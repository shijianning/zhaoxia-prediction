"""启发式朝霞晚霞评分模型（气象规则）。

火烧云(朝霞/晚霞)形成的关键条件，来源于大气光学与气象学经验：
1. 中高云(高积云、卷云)适中 —— 云层承接并散射红橙光，太少平淡、太多阴沉；
2. 低云不能太厚 —— 否则遮挡地平线视线；
3. 湿度适中 —— 一定水汽利于色彩，近饱和则成雾霾；
4. 无降水 —— 有雨基本没戏；
5. 风速适中 —— 静风偏灰、大风扬尘；
6. 大气通透(气溶胶光学厚度 AOD 低) —— 天空越通透，火烧云颜色越鲜艳；
7. 水平能见度高 —— 低空无霾/雾，红橙光衰减少、色彩更艳。

第 1/2/3/4/5 点对应 sunset-prediction 的 5 因子加权模型
（云型/能见度/湿度/降水/总云量）；第 6/7 点吸收 chromasky 的 AOD 思想与
能见度因子，二者共同构成"通透类"打分，合计权重约 28%。

**第 8/9 点补充"方向性"**（见 transect.py）：上面 1~7 点用的都是"观测点头顶
的平均云量"，等于假设天空各方向均匀。但朝霞晚霞是有方向性的光学现象——
8. 剖面云边界 —— 沿太阳方位找云层边缘的位置。云层在约 400km 处到达边缘时，
   边缘外的晴空让低角度阳光从云层下方斜射进来，把近处云幕的下表面整体点亮，
   这是最壮观火烧云的成因（chromasky 权重最高的因子）；
9. 太阳方位遮挡 —— 太阳方向近场(0~150km)的低云量。这一段的低云会直接把
   低角度光路切断，比全天空平均低云更能反映真实遮挡。

权重设计：原有 7 因子权重整体缩放到 0.84，8/9 两点合计 0.16。
`rule_score` 会**动态归一化**——当剖面数据不可用（网络失败或未启用）时，
权重自动回落到原有 7 因子，结果与未启用剖面时逐位一致，保证优雅降级。

输出 0-100 分，以及各因素拆解，用于报告展示与 ML 弱监督标签。
"""


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ---- 权重表 ---------------------------------------------------------------
# 原 7 因子（合计 1.00），来自各开源项目的经验标定
_BASE_WEIGHTS = {
    "云结构": 0.30,
    "低云遮挡": 0.12,
    "湿度": 0.12,
    "降水": 0.10,
    "风速": 0.08,
    "通透度": 0.15,
    "能见度": 0.13,
}

# 剖面因子权重（合计 0.16）。原有 7 因子同时乘以 (1 - 0.16) 保持总和为 1。
_TRANSECT_WEIGHTS = {
    "剖面云边界": 0.09,
    "太阳方位遮挡": 0.07,
}

_PROFILE_DECAY = 1.0 - sum(_TRANSECT_WEIGHTS.values())   # 0.84

# 云层边缘效应的两个关键距离（公里）：
# 400km 处存在边缘最优（低角度阳光从边缘下方照亮近处云底），
# 超过 500km 视为"云幕铺满全程、没有边缘"。
OPTIMAL_BOUNDARY_KM = 400.0
MAX_BOUNDARY_KM = 500.0
# 云量低于该值即视为云层边缘 / 晴空
CLOUD_EDGE_THRESHOLD = 10.0

# 太阳方位近场低云的扣分斜率。近场低云直接切断光路，
# 比全天空平均低云（斜率 0.55）更决定性，故取更陡的 0.9。
AZIMUTH_BLOCK_SLOPE = 0.9


def transparency_score(aod):
    """把气溶胶光学厚度(AOD)映射为通透度评分。

    AOD 越低天空越通透、火烧云越鲜艳；AOD 高则颜色发灰。
    参考 SunsetBot 的 AOD 分级（约 0.6+ 即"污"）。
    aod 为 None 时给中性分（无数据不奖惩）。
    """
    if aod is None:
        return 70.0
    if aod <= 0.15:
        return 100.0
    if aod <= 0.3:
        return 88.0
    if aod <= 0.45:
        return 72.0
    if aod <= 0.6:
        return 55.0
    if aod <= 0.8:
        return 35.0
    return 18.0


def visibility_score(vis):
    """把水平能见度(米)映射为通透度评分。

    能见度越高，低空霾/雾越少，红橙光衰减越少、火烧云越鲜艳。
    Open-Meteo 的 visibility 上限约 24km，故阈值按此标定。
    vis 为 None 时给中性分。
    """
    if vis is None:
        return 70.0
    if vis >= 20000:
        return 100.0
    if vis >= 15000:
        return 90.0
    if vis >= 10000:
        return 78.0
    if vis >= 6000:
        return 60.0
    if vis >= 3000:
        return 38.0
    return 18.0


def boundary_distance_score(profile, distances):
    """剖面云边界因子：沿太阳方位找到云层边缘的位置，按经验曲线打分。

    profile:   沿太阳方位的中高云量序列（%），对应 distances 上的采样点。
    distances: 各采样点到观测点的距离（公里）。

    物理依据是"云层边缘效应"：云层在约 400km 处到达边缘时，边缘以外的晴空
    让低角度阳光从云层下方斜射进来，把近处云幕的下表面整体点亮；云幕一直铺到
    天边（不见边缘）或近处就没云（无幕布承接），都烧不起来。

    曲线（沿用 chromasky 的经验标定）：
        0km → 0 分；400km → 满分 100；400~500km 线性衰减到 0；≥500km → 0 分。

    与"云结构"的分工：本因子只回答"边缘在哪"，不回答"有没有云"。
    因此当射线起点（头顶）就没有中高云、无从谈边缘时，返回 (None, None)，
    由调用方把这一项**整体移出加权**——注意不是给个中间分，因为带权重的
    中间分依然会牵动总分，那不叫"不表态"。这样"没云 = 烧不起来"的判断
    完全交给"云结构"因子（它看的是全天空总云量），避免同一件事被扣两次分。

    返回 (boundary_km, score)；不适用时返回 (None, None)。
    """
    if not profile:
        return None, None

    # 头顶没有承光的云幕 → 本因子无从判断，整体移出加权
    if profile[0] < CLOUD_EDGE_THRESHOLD:
        return None, None

    edge_km = None
    for dist, value in zip(distances, profile):
        if dist <= 0:
            continue
        if value < CLOUD_EDGE_THRESHOLD:
            edge_km = float(dist)
            break

    if edge_km is None:
        # 全程都有云 —— 没有边缘，缺少"从下方打光"的通道
        return MAX_BOUNDARY_KM, 0.0

    if edge_km <= OPTIMAL_BOUNDARY_KM:
        # 边缘越近，近处可供承接阳光的云幕越少
        score = edge_km / OPTIMAL_BOUNDARY_KM * 100.0
    else:
        score = (1.0 - (edge_km - OPTIMAL_BOUNDARY_KM)
                 / (MAX_BOUNDARY_KM - OPTIMAL_BOUNDARY_KM)) * 100.0
    return edge_km, round(_clamp(score), 1)


def azimuth_block_score(near_low_cloud):
    """太阳方位遮挡因子：太阳方向近场低云量越低越好。

    near_low_cloud: 0~150km 范围内的低云量均值（%）。无数据时返回 None，
    由调用方把这一项整体移出加权（而不是给中间分）。

    日出/日落时阳光几乎水平地射来，近场这段的低云会把整条光路切断，
    因此它比"全天空平均低云"更能反映真实遮挡，扣分也更陡。
    """
    if near_low_cloud is None:
        return None
    return round(_clamp(100.0 - float(near_low_cloud) * AZIMUTH_BLOCK_SLOPE), 1)


def rule_score(f):
    """计算规则评分。

    f: 特征字典（见 features.py）。若含 "transect"（见 transect.py），
       则额外计入剖面云边界与太阳方位遮挡两个方向性因子。

    返回 (score_0_100, breakdown_dict)。
    """
    high = f.get("cloud_high", 0.0)
    mid = f.get("cloud_mid", 0.0)
    low = f.get("cloud_low", 0.0)
    rh = f.get("humidity", 0.0)
    wind = f.get("wind", 0.0)
    precip = f.get("precip", 0.0)
    aod = f.get("aod")
    vis = f.get("visibility")

    breakdown = {}

    # 1) 云结构：中高云含量适中最佳(约 50-60%)
    upper = min(high + mid, 100.0)
    if upper < 8:
        cloud_score = 12.0        # 几乎无云，平淡
    elif upper > 92:
        cloud_score = 18.0        # 全阴，无光
    else:
        cloud_score = _clamp(100 - abs(upper - 55) * 1.4)
    breakdown["云结构"] = round(cloud_score, 1)

    # 2) 低云遮挡
    low_view = _clamp(100 - low * 0.55)
    breakdown["低云遮挡"] = round(low_view, 1)

    # 3) 湿度：55% 附近最佳
    hum_score = _clamp(100 - abs(rh - 55) * 1.6)
    breakdown["湿度"] = round(hum_score, 1)

    # 4) 降水
    if precip > 0.2:
        rain_score = 0.0
    elif precip > 0.05:
        rain_score = 30.0
    else:
        rain_score = 100.0
    breakdown["降水"] = round(rain_score, 1)

    # 5) 风速
    if wind < 2:
        wind_score = 85.0
    elif wind < 6:
        wind_score = 100.0
    elif wind < 10:
        wind_score = 70.0
    else:
        wind_score = 40.0
    breakdown["风速"] = round(wind_score, 1)

    # 6) 大气通透度(AOD)
    trans_score = transparency_score(aod)
    breakdown["通透度"] = round(trans_score, 1)

    # 7) 水平能见度（融合 sunset-prediction 能见度因子）
    vis_score = visibility_score(vis)
    breakdown["能见度"] = round(vis_score, 1)

    # 8/9) 方向性因子（需 transect.py 的剖面数据）
    # 权重动态归一化：剖面数据缺失时自动回落到原有 7 因子，
    # 使"未启用/取数失败"的结果与升级前逐位一致。
    weights = dict(_BASE_WEIGHTS)
    transect_info = f.get("transect") or {}
    boundary = transect_info.get("boundary_score")
    block = transect_info.get("block_score")
    if boundary is not None or block is not None:
        weights = {k: v * _PROFILE_DECAY for k, v in weights.items()}
        if boundary is not None:
            breakdown["剖面云边界"] = round(float(boundary), 1)
            weights["剖面云边界"] = _TRANSECT_WEIGHTS["剖面云边界"]
        if block is not None:
            breakdown["太阳方位遮挡"] = round(float(block), 1)
            weights["太阳方位遮挡"] = _TRANSECT_WEIGHTS["太阳方位遮挡"]

    total_weight = sum(weights.values())
    score = sum(weights[k] * breakdown[k] for k in weights) / total_weight
    breakdown["综合"] = round(_clamp(score), 1)
    return breakdown["综合"], breakdown


def grade_of(score):
    """把 0-100 分映射为中文评级。"""
    if score >= 75:
        return "优秀"
    if score >= 60:
        return "良好"
    if score >= 45:
        return "一般"
    return "平淡"


def vividness_of(score):
    """把 0-100 分映射为摄影圈通用的火烧云"鲜艳度"分级（对齐 SunsetBot）。"""
    if score >= 90:
        return "世纪大烧"
    if score >= 80:
        return "优质大烧"
    if score >= 70:
        return "大烧"
    if score >= 60:
        return "中烧"
    if score >= 45:
        return "小烧"
    return "微烧/无"


def chroma_index(score):
    """把 0-100 分映射为 0-10 的「鲜艳度指数」（对齐 chromasky 的 ChromaSky™ 指数）。

    标准化到 0-10 便于跨城市、跨日期横向对比，也便于全国地图统一色阶。
    """
    return round(_clamp(score) / 10.0, 1)


def factor_note(breakdown):
    """找出最拖后腿的因素，生成一句人话提示。"""
    # 排除综合分
    items = {k: v for k, v in breakdown.items() if k != "综合"}
    worst_key = min(items, key=items.get)
    worst_val = items[worst_key]
    mapping = {
        "云结构": "云层结构不佳（中高云太少或过厚）",
        "低云遮挡": "低云较多，遮挡视线",
        "湿度": "空气湿度不理想",
        "降水": "有降水可能，云层不易显色",
        "风速": "风速不理想",
        "通透度": "大气浑浊（气溶胶偏高），颜色易发灰",
        "能见度": "低空能见度不足（有霾/雾），红橙光衰减明显",
        "剖面云边界": "太阳方向没有理想的云层边缘，光难以从云底下方打进来",
        "太阳方位遮挡": "太阳方向近处低云偏多，会挡住低角度阳光",
    }
    if worst_val >= 80:
        return "各因素配合良好，具备出霞条件"
    return mapping.get(worst_key, worst_key)
