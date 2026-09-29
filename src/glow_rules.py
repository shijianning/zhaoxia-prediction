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
能见度因子，二者共同构成"通透类"打分。

**方向性因子**（见 transect.py）：上面 1~7 点用的都是"观测点头顶的平均云量"，
等于假设天空各方向均匀。但朝霞晚霞是有方向性的光学现象——
8. 剖面云边界 —— 沿太阳方位找云层边缘的位置。云层在约 400km 处到达边缘时，
   边缘外的晴空让低角度阳光从云层下方斜射进来，把近处云幕的下表面整体点亮，
   这是最壮观火烧云的成因（chromasky 权重最高的因子）；
9. 太阳方位遮挡 —— 太阳方向近场(0~150km)的低云量。这一段的低云会直接把
   低角度光路切断，比全天空平均低云更能反映真实遮挡；
10. 地形遮蔽 —— 太阳方位的地平线仰角（PVGIS 90m DEM）。西边有山时，
    地平线被抬高，低角度火烧云所在的天空可见范围被压缩。

### 缺失数据的处理约定（重要）

规则引擎对"取不到数据"只有一个语义：**把该因子整个移出加权，然后对剩余权重
重新归一化**。绝不用"中间分"顶替 —— 带权重的中间分依然会牵动总分，那不叫不表态。

`rule_score` 因此写成"先收集可用因子，再按可用集合归一化权重"的形式：
- 7 个基础因子全可用且无方向性数据时，结果与升级前**逐位一致**；
- AOD 或能见度缺失时，它们的权重（0.15 / 0.13）被剔除，其余因子按比例放大；
- 方向性因子缺失时同理。

输出 0-100 分，以及各因素拆解，用于报告展示与 ML 弱监督标签。
"""


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ---- 权重表 ---------------------------------------------------------------
# 7 个基础因子（合计 1.00），来自各开源项目的经验标定
_BASE_WEIGHTS = {
    "云结构": 0.30,
    "低云遮挡": 0.12,
    "湿度": 0.12,
    "降水": 0.10,
    "风速": 0.08,
    "通透度": 0.15,
    "能见度": 0.13,
}

# 方向性因子（合计 0.16）。基础因子权重会按 (1 - 可用方向性权重和) 等比缩放。
_TRANSECT_WEIGHTS = {
    "剖面云边界": 0.08,
    "太阳方位遮挡": 0.06,
    "地形遮蔽": 0.02,
}

# 方向性因子全部可用时，基础因子的缩放系数（0.84），仅用于文档与测试参考
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

# 地形遮蔽：地平线仰角每升高 1° 扣多少分。
# 2° 山体 → 82 分，5° → 55 分，10° → 10 分。
TERRAIN_SLOPE = 9.0

# 沙尘分型阈值：dust/pm10 高于此值视为沙尘主导
DUST_RATIO_THRESHOLD = 0.5


# ---- 通透度（AOD）---------------------------------------------------------
def _clean_branch(aod):
    """清洁气溶胶：AOD 越低越通透，单调递减。"""
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


def _dust_branch(aod):
    """沙尘主导：关系是**非单调**的，AOD≈0.25 附近反而最优。

    沙尘以粗颗粒为主，前向散射强，适量的沙尘等于给天空加了一层"暖色滤镜"，
    会强化红橙色调；但过浓则整体发黄发灰、把色彩压平。
    这与清洁气溶胶"越少越好"的单调关系相反，必须分型处理。
    """
    if aod <= 0.1:
        return 55.0        # 太干净，缺少散射介质
    if aod <= 0.25:
        return 55.0 + (aod - 0.1) / 0.15 * 45.0   # 0.10→55, 0.25→100
    if aod <= 0.45:
        return 100.0 - (aod - 0.25) / 0.20 * 20.0  # 0.25→100, 0.45→80
    if aod <= 0.8:
        return 80.0 - (aod - 0.45) / 0.35 * 30.0   # 0.45→80, 0.80→50
    if aod <= 1.3:
        return 50.0 - (aod - 0.8) / 0.5 * 30.0     # 0.80→50, 1.30→20
    return 15.0


def dust_ratio_of(f):
    """从特征里估算沙尘占比（dust/pm10）。取不到返回 None。"""
    dust = f.get("dust")
    pm10 = f.get("pm10")
    if dust is None or pm10 is None or pm10 <= 0:
        return None
    return float(dust) / float(pm10)


def transparency_score(aod, dust_ratio=None):
    """把气溶胶光学厚度(AOD)映射为通透度评分。

    AOD 越低天空越通透、火烧云越鲜艳（清洁支）；
    但沙尘主导时关系转为非单调，峰值在 AOD≈0.25（沙尘支）。

    aod 为 None 时返回 None —— 由调用方把该因子**整体移出加权**，
    而不是给一个"中间分"：带权重的中间分依然会牵动总分。
    """
    if aod is None:
        return None
    aod = float(aod)
    if dust_ratio is not None and dust_ratio >= DUST_RATIO_THRESHOLD:
        return round(_clamp(_dust_branch(aod)), 1)
    return round(_clamp(_clean_branch(aod)), 1)


def washout_factor(precip_24h):
    """窗口前 24h 降水对气溶胶的**湿沉降/吸湿增长**调制系数。

    文献经验（用于修正"预报 AOD 未反映降水后处理"的偏差）：
    - 大雨（≥10mm）冲刷气溶胶约 50%，通透度实际优于预报值 → 系数 < 1；
    - 中雨（2~10mm）轻度冲刷 → 略小于 1；
    - 微量降水（0~2mm）反而因吸湿增长使 PM2.5 上升 10~20% → 系数略大于 1。

    precip_24h 为 None（无数据）时返回 1.0（不干预）。
    """
    if precip_24h is None:
        return 1.0
    if precip_24h >= 10.0:
        return 0.75
    if precip_24h >= 2.0:
        return 0.92
    if precip_24h > 0.0:
        return 1.08
    return 1.0


def visibility_score(vis):
    """把水平能见度(米)映射为通透度评分。

    能见度越高，低空霾/雾越少，红橙光衰减越少、火烧云越鲜艳。
    Open-Meteo 的 visibility 上限约 24km，故阈值按此标定。
    vis 为 None 时返回 None（移出加权）。
    """
    if vis is None:
        return None
    vis = float(vis)
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


def terrain_score(terrain_deg):
    """地形遮蔽因子：太阳方位的地平线仰角越低越好。

    terrain_deg: 太阳方位上由地形抬高的地平线仰角（度，PVGIS 90m DEM）。
    观察者能看到的低空天空被这个角度切掉一块 —— 而低角度火烧云恰恰就在
    那片天空里。山区城市（重庆、贵阳等）西侧群山会把这个值抬到 2° 以上。

    无数据时返回 None（整体移出加权）。
    """
    if terrain_deg is None:
        return None
    return round(_clamp(100.0 - float(terrain_deg) * TERRAIN_SLOPE), 1)


def precip_score(precip, precip_24h=None):
    """降水因子：当前窗口无雨是前提；前 24h 的微量降水另作轻罚。

    大雨的"洗尘红利"不在这里重复加分 —— 它通过 washout_factor 作用在
    AOD 上（物理上更准确：降水改变的是气溶胶，而不是降水本身的好坏）。
    """
    if precip > 0.2:
        return 0.0
    if precip > 0.05:
        return 30.0
    if precip_24h is not None and 0.0 < precip_24h < 5.0:
        return 88.0     # 微量降水：吸湿增长，通透度略降
    return 100.0


def rule_score(f):
    """计算规则评分。

    f: 特征字典（见 features.py）。若含 "transect"（见 transect.py），
       则额外计入方向性因子（剖面云边界 / 太阳方位遮挡 / 地形遮蔽）。

    缺失值语义：值为 None 的因子**不参与加权**，其余因子权重按可用集合
    重新归一化（见模块 docstring）。当 7 个基础因子齐全且无方向性数据时，
    结果与引入方向性之前的版本逐位一致。

    返回 (score_0_100, breakdown_dict)。
    """
    high = f.get("cloud_high")
    mid = f.get("cloud_mid")
    low = f.get("cloud_low")
    rh = f.get("humidity", 0.0)
    wind = f.get("wind", 0.0)
    precip = f.get("precip", 0.0)

    # scores 只收集"有数据"的因子；缺失的直接不进来
    scores = {}

    # 1) 云结构：中高云含量适中最佳(约 55%)。需要中云与高云两维都有值
    if high is not None and mid is not None:
        upper = min(high + mid, 100.0)
        if upper < 8:
            scores["云结构"] = 12.0        # 几乎无云，平淡
        elif upper > 92:
            scores["云结构"] = 18.0        # 全阴，无光
        else:
            scores["云结构"] = _clamp(100 - abs(upper - 55) * 1.4)

    # 2) 低云遮挡
    if low is not None:
        scores["低云遮挡"] = _clamp(100 - low * 0.55)

    # 3) 湿度：55% 附近最佳
    scores["湿度"] = _clamp(100 - abs(rh - 55) * 1.6)

    # 4) 降水（含前 24h 微量降水的轻罚）
    scores["降水"] = precip_score(precip, f.get("precip_24h"))

    # 5) 风速
    if wind < 2:
        scores["风速"] = 85.0
    elif wind < 6:
        scores["风速"] = 100.0
    elif wind < 10:
        scores["风速"] = 70.0
    else:
        scores["风速"] = 40.0

    # 6) 大气通透度(AOD)：先按前 24h 降水做湿沉降/吸湿修正，再分型打分
    aod = f.get("aod")
    if aod is not None:
        aod_eff = float(aod) * washout_factor(f.get("precip_24h"))
        s = transparency_score(aod_eff, dust_ratio_of(f))
        if s is not None:
            scores["通透度"] = s

    # 7) 水平能见度
    vis = f.get("visibility")
    if vis is not None:
        s = visibility_score(vis)
        if s is not None:
            scores["能见度"] = s

    # 8/9/10) 方向性因子（需 transect.py 的剖面 + 地形数据）
    tr = f.get("transect") or {}
    for name, key in (("剖面云边界", "boundary_score"),
                      ("太阳方位遮挡", "block_score"),
                      ("地形遮蔽", "terrain_score")):
        v = tr.get(key)
        if v is not None:
            scores[name] = float(v)

    # ---- 加权：基础因子按可用集合等比缩放，腾出方向性因子的权重 ----
    base_avail = [k for k in scores if k in _BASE_WEIGHTS]
    prof_avail = [k for k in scores if k in _TRANSECT_WEIGHTS]
    prof_total = sum(_TRANSECT_WEIGHTS[k] for k in prof_avail)
    base_sum = sum(_BASE_WEIGHTS[k] for k in base_avail)

    weights = {}
    if base_sum > 0:
        scale = (1.0 - prof_total) / base_sum
        for k in base_avail:
            weights[k] = _BASE_WEIGHTS[k] * scale
    for k in prof_avail:
        weights[k] = _TRANSECT_WEIGHTS[k]

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 50.0, {"综合": 50.0}
    score = sum(weights[k] * scores[k] for k in weights) / total_weight

    breakdown = {k: round(float(v), 1) for k, v in scores.items()}
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
        "地形遮蔽": "太阳方向有山体挡住低空，可见的霞光范围被压缩",
    }
    if worst_val >= 80:
        return "各因素配合良好，具备出霞条件"
    return mapping.get(worst_key, worst_key)
