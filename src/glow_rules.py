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

输出 0-100 分，以及各因素拆解，用于报告展示与 ML 弱监督标签。
"""


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


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


def rule_score(f):
    """计算规则评分。

    f: 特征字典（见 features.py）。
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

    score = (
        0.30 * cloud_score
        + 0.12 * low_view
        + 0.12 * hum_score
        + 0.10 * rain_score
        + 0.08 * wind_score
        + 0.15 * trans_score
        + 0.13 * vis_score
    )
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
    }
    if worst_val >= 80:
        return "各因素配合良好，具备出霞条件"
    return mapping.get(worst_key, worst_key)
