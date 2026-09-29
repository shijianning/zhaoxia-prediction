"""太阳方位剖面分析 —— 把"看头顶云量"升级为"看朝向太阳的那条线"。

原有的规则引擎只看观测点头顶 ±1 小时的平均云量，等于假设天空各方向均匀。
但朝霞/晚霞本质上是**有方向性的光学现象**，三个被忽略的几何事实：

1. **光来自太阳所在的方位。** 日出在东方、日落在西方，真正决定成败的是
   太阳那半个天空，而不是全天空平均。日出方位的低云会把低角度阳光整条切断。

2. **承光的"幕布"在几十到几百公里之外。** 火烧云不是头顶那朵云在发光，
   而是太阳方向 100~400km 处的中高云被从下方照亮。

3. **经典机制是"云层边缘效应"。** 当云层在约 400km 处到达边缘时，
   边缘以外的晴空让低角度阳光从云层下方斜射进来，把近处云幕的下表面整体
   点亮 —— 这是最壮观的火烧云成因。反过来，云一直铺到天边（无边缘）
   或近处就没云（无幕布），都烧不起来。

本模块沿日出/日落方位角拉一条 0~500km 的剖面。关键在于这**不需要额外的数据
成本**：Open-Meteo 支持单次请求多个坐标，早、晚两条射线各 17 个点（共 34 点）
合并成一次调用，仍然是免费无 Key。

产出三个新因子（打分曲线在 glow_rules.py 中，便于统一校准）：

- `boundary_score`     剖面云边界：沿太阳方位找到第一处"云量近乎为零"的距离，
  按经验曲线打分（0km→0、400km→满分、≥500km→0，峰值在 400km）。
- `block_score`        太阳方位遮挡：0~150km 这段"地平线附近视野"的低云量均值，
  越低越好。比全天空平均低云更贴近真实的遮光效果。
- `terrain_score`      地形遮蔽：太阳方位上的地平线仰角（PVGIS 90m DEM，见 horizon.py），
  山区城市西侧群山会把可见的低空天空切掉一块。

同时产出两个**光照几何诊断量**（不参与打分，供报告解释）：

- `light_horizon_km`   阳光与地面相切的距离。日落瞬间约 185km ——
  比这更近处光线仍在地面之下，那里的云照不到光。这是"近场低云遮挡"
  取 0~150km 的几何依据。
- `light_path_km`      剖面各距离处阳光离地的高度。

任何一步失败都返回 `{}`，调用方自动退回原有 7 因子评分，结果与未启用时完全一致。
"""

from . import glow_rules
from . import horizon
from . import solar
from . import weather

# 剖面采样距离（公里）。0 = 观测点本身。
# 最远取到 500km：再远的光线已不可能以日出/日落那样的低角度越过地球曲率
# 照到本地云底（见 solar.shadow_height_deg）。
TRANSECT_DISTANCES_KM = [
    0, 20, 40, 60, 80, 100, 130, 160, 190, 220, 260, 300, 340, 380, 420, 460, 500,
]

# "近场"范围：低于此距离的低云会直接切断通往太阳方向的低角度光路。
# 1km 高的低云地平线距离约 113km，取 150km 留出余量。
NEAR_FIELD_KM = 150.0

# 用于说明"哪个高度还能被照亮"的代表性云层高度（公里）
_REPORT_LAYER_KM = 3.0


def enabled(cfg, scope="single"):
    """配置开关。

    transect.enabled       —— 单城路径（daily_run / 报告）是否启用，默认开。
    transect.national_map  —— 全国地图是否同样启用，默认开。
                              全国地图每城多一次请求，若被限流可单独关掉。
    """
    section = cfg.get("transect") or {}
    if scope == "national":
        return bool(section.get("enabled", True)) and bool(
            section.get("national_map", True)
        )
    return bool(section.get("enabled", True))


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fill_profile(vals):
    """把剖面里缺失的点用同一条射线上已知值的均值补齐。

    直接填 0 会伪造出"云层边缘"（云量突降为零），让边界因子误判；
    用均值填补只改变幅度、不改变形状，对边界位置判断更安全。
    全部缺失时返回 None，表示该因子不可用。
    """
    known = [v for v in vals if v is not None]
    if not known:
        return None
    fallback = sum(known) / len(known)
    return [v if v is not None else fallback for v in vals]


def _event_times(data):
    """从主预报数据里取出每天的日出/日落本地时刻。

    返回 {(date, window): "YYYY-MM-DDTHH:MM"}。
    """
    daily = data.get("daily", {})
    out = {}
    for i, day in enumerate(daily.get("time", [])):
        for window, key in (("morning", "sunrise"), ("evening", "sunset")):
            arr = daily.get(key) or []
            if i < len(arr) and arr[i]:
                out[(day, window)] = arr[i]
    return out


def _profile_at(point, hour_key, var):
    """取某个剖面点上、指定小时（含前后各 1 小时）的变量均值。"""
    hourly = point.get("hourly", {})
    times = hourly.get("time", [])
    vals = hourly.get(var, [])
    if not vals:
        return None
    idx = None
    for i, t in enumerate(times):
        if t[:13] == hour_key:
            idx = i
            break
    if idx is None:
        return None
    lo = max(0, idx - 1)
    hi = min(len(vals) - 1, idx + 1)
    return _mean(vals[lo:hi + 1])


def _representative_azimuths(city, events, utc_offset):
    """为朝、晚两个窗口各算一个代表方位角。

    7 天之内日出/日落的方位角移动不足 1°，剖面选点用同一条射线完全够用；
    这样早、晚两条射线才能合并成一次多坐标请求（成本不翻倍）。
    """
    days = sorted({d for (d, _) in events})
    if not days:
        return {}
    mid_day = days[len(days) // 2]

    out = {}
    for window in ("morning", "evening"):
        iso = events.get((mid_day, window))
        if not iso:
            continue
        pos = solar.event_position(
            city["latitude"], city["longitude"], iso, utc_offset
        )
        out[window] = pos
    return out


def compute_transect(cfg, data, days=None):
    """拉取沿日出/日落方位角的剖面，返回 {(date, window): factors}。

    cfg:  完整配置；用 cfg["city"] 取经纬度与时区。
    data: 主预报的原始返回（提供日出日落时刻与 utc_offset_seconds）。
    days: 预测天数，默认取 cfg["forecast"]["days"]。

    失败（网络异常 / 返回结构不符 / 全无云量数据）时返回 {}。
    """
    city = cfg["city"]
    lat, lon = city["latitude"], city["longitude"]
    utc_offset = int(data.get("utc_offset_seconds") or 0)

    events = _event_times(data)
    if not events:
        return {}

    azimuths = _representative_azimuths(city, events, utc_offset)
    if not azimuths:
        return {}

    # 地形剖面：每城只拉一次（剖面与方位无关，方位只用于插值），
    # 避免"最先算到的那个窗口"独自承担网络冷启动失败而留下空洞。
    horizon_profile = None
    if horizon.enabled(cfg):
        try:
            horizon_profile = horizon.fetch_horizon(cfg, lat, lon)
        except Exception:
            horizon_profile = None

    # ---- 组装两条射线：每个窗口 17 个点，合并为一次请求 ----
    rays = []          # [(window, [距离...])]
    all_lats, all_lons = [], []
    for window in ("morning", "evening"):
        pos = azimuths.get(window)
        if pos is None:
            continue
        lat_list, lon_list = [], []
        for dist in TRANSECT_DISTANCES_KM:
            plat, plon = solar.destination_point(lat, lon, pos["azimuth"], dist)
            lat_list.append(plat)
            lon_list.append(plon)
        rays.append((window, lat_list, lon_list))
        all_lats.extend(lat_list)
        all_lons.extend(lon_list)

    if not all_lats:
        return {}

    try:
        payload = weather.get_transect(cfg, all_lats, all_lons, days=days)
    except Exception:
        return {}

    if not isinstance(payload, list) or len(payload) != len(all_lats):
        return {}

    # 把扁平响应切回每条射线
    per_window_points = {}
    cursor = 0
    for window, lat_list, _ in rays:
        n = len(lat_list)
        per_window_points[window] = payload[cursor:cursor + n]
        cursor += n

    # ---- 逐个日期/窗口计算因子 ----
    out = {}
    for (date, window), iso in events.items():
        points = per_window_points.get(window)
        if not points:
            continue

        hour_key = iso[:13]
        hcc = _fill_profile(
            [_profile_at(p, hour_key, "cloud_cover_high") for p in points]
        )
        lcc = _fill_profile(
            [_profile_at(p, hour_key, "cloud_cover_low") for p in points]
        )
        mcc = _fill_profile(
            [_profile_at(p, hour_key, "cloud_cover_mid") for p in points]
        )
        if hcc is None and lcc is None:
            continue

        # 承光"幕布"取中云 + 高云：日落时这两层的云底都能被低角度阳光照亮
        # （见 solar.shadow_height_deg），只用高云会漏掉"中云满布但无卷云"的天气。
        if hcc is not None and mcc is not None:
            canopy = [min(a + b, 100.0) for a, b in zip(hcc, mcc)]
        else:
            canopy = hcc if hcc is not None else mcc

        boundary_km, boundary_score = (None, None)
        if canopy is not None:
            boundary_km, boundary_score = glow_rules.boundary_distance_score(
                canopy, TRANSECT_DISTANCES_KM
            )

        near_low = None
        block_score = None
        if lcc is not None:
            near_vals = [
                v for d, v in zip(TRANSECT_DISTANCES_KM, lcc) if d <= NEAR_FIELD_KM
            ]
            near_low = _mean(near_vals)
            block_score = glow_rules.azimuth_block_score(near_low)

        pos = azimuths.get(window) or {}
        az_val = pos.get("azimuth")
        elev_val = pos.get("elevation")

        # 地形遮蔽：在已拉取的剖面上按该窗口的太阳方位插值（无额外请求）
        terrain_deg = None
        if horizon_profile is not None and az_val is not None:
            try:
                terrain_deg = horizon.interpolate(horizon_profile, az_val)
            except Exception:
                terrain_deg = None

        out[(date, window)] = {
            "azimuth": round(pos.get("azimuth", 0.0), 1),
            "sun_elev": round(pos.get("elevation", 0.0), 2),
            "boundary_km": boundary_km,
            "boundary_score": boundary_score,
            "near_low_cloud": None if near_low is None else round(near_low, 1),
            "block_score": block_score,
            # 地形遮蔽（PVGIS 90m DEM）
            "terrain_deg": None if terrain_deg is None else round(float(terrain_deg), 2),
            "terrain_score": glow_rules.terrain_score(terrain_deg),
            # 保留剖面原始数据，供报告可视化与后续调参
            "ray_canopy": None if canopy is None else [round(v, 1) for v in canopy],
            "ray_hcc": None if hcc is None else [round(v, 1) for v in hcc],
            "ray_lcc": None if lcc is None else [round(v, 1) for v in lcc],
            "ray_mcc": None if mcc is None else [round(v, 1) for v in mcc],
            "distances_km": list(TRANSECT_DISTANCES_KM),
            # 该高度云层还能接光所允许的最大太阳下沉角（说明用）
            "shadow_deg": round(solar.shadow_height_deg(_REPORT_LAYER_KM, lat), 2),
            # ---- 光照几何 ----
            # 阳光与地面相切的距离：比它更近处光线仍在地面之下，云照不到光。
            # 日落瞬间约 185km —— 这就是"近场低云遮挡"取 0~150km 的几何依据。
            "light_horizon_km": round(solar.light_horizon_km(elev_val, lat), 1),
            # 剖面各距离处阳光离地的高度（负值 = 仍在地面之下）
            "light_path_km": [
                round(solar.light_path_height_km(d, elev_val, lat), 2)
                for d in TRANSECT_DISTANCES_KM
            ],
        }
    return out


def describe(factors):
    """把剖面因子翻译成一句人话，供报告/推送使用。"""
    if not factors:
        return ""
    parts = []
    az = factors.get("azimuth")
    if az is not None:
        parts.append(f"太阳方位 {az:.0f}°")
    km = factors.get("boundary_km")
    if km is not None:
        if km >= 500:
            parts.append("云幕铺满全程(缺边缘)")
        elif km <= 0:
            parts.append("太阳方向无云幕")
        else:
            parts.append(f"云层边缘约在 {km:.0f}km 外")
    near = factors.get("near_low_cloud")
    if near is not None:
        parts.append(f"近场低云 {near:.0f}%")
    terr = factors.get("terrain_deg")
    if terr is not None and terr > 0.05:
        parts.append(f"地形仰角 {terr:.1f}°")
    return "，".join(parts)
