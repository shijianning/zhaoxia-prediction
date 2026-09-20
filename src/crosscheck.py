"""多气象模型交叉验证模块。

用多个独立的全球气象模式（ECMWF IFS / NOAA GFS / DWD ICON）对同一地点、
同一日出/日落窗口各自计算规则评分，并给出模型间一致性/置信度，
用于佐证主预测：结论一致时可信度高，分歧大时说明该时段云况不稳定、预报不确定。

这是"用不同预测程序互相印证"的免费实现（无需 API Key）。
"""
from . import weather
from .features import extract_day_features
from .glow_rules import grade_of, rule_score

# 三个来自不同国家气象机构的独立全球模式（云量数据经实测均可用）
CROSSCHECK_MODELS = ["gfs_global", "icon_global", "gem_global"]

MODEL_LABEL = {
    "gfs_global": "GFS(美)",
    "icon_global": "ICON(德)",
    "gem_global": "GEM(加)",
}


def _model_hourly(multi_data, model):
    """把多模型返回的 `<变量>_<模型>` 字段还原为单模型结构。"""
    suffix = "_" + model
    hourly = multi_data.get("hourly", {})
    new_hourly = {"time": hourly.get("time", [])}
    for key, vals in hourly.items():
        if key.endswith(suffix):
            new_hourly[key[: -len(suffix)]] = vals
    return new_hourly


def _valid_hourly(hourly):
    """判断该模型是否真有云量数据（部分模型可能返回全空）。"""
    cc = hourly.get("cloud_cover")
    return bool(cc) and any(v is not None for v in cc)


def confidence_of(spread):
    """根据模型间分差给出置信度。"""
    if spread <= 10:
        return "高"
    if spread <= 25:
        return "中"
    return "低"


def run_geovisearth_crosscheck(cfg, days=None):
    """第 4 方第三方对比：星图云(GeoVisEarth)官方「火烧云预报API」。

    这是对 SunsetBot / 莉景天气 这类「无公开 API」小程序的最佳替代：
    星图云开放平台提供官方朝霞/晚霞预报接口（火烧云质量 + 等级，未来 3 天，
    每日日出/日落两次），需申请 token。接口文档：
    https://open.geovisearth.com/support/document?docId=757&detail=client

    当前为「占位桩」：未在 config.yaml 填入 token 与 productCode/dataCode/meteCode
    时直接返回空 dict（不参与对比）；填入并 enabled=true 后，在下方 TODO 处
    按官方返回结构解析「火烧云等级」，映射为 0-100 分挂到报告上。

    返回 {(date, window): info}，与 run_crosscheck 结构一致。
    """
    gv = (cfg.get("third_party") or {}).get("geovisearth") or {}
    if not gv.get("enabled") or not gv.get("token"):
        return {}
    token = gv["token"]
    base_url = gv.get("base_url", "https://api.open.geovisearth.com/v2/glow/fc/idxV2")
    product_code = gv.get("productCode", "")
    data_code = gv.get("dataCode", "")
    mete_code = gv.get("meteCode", "")

    if not (product_code and data_code and mete_code):
        # 参数不全，视为未启用（避免发出必错请求）
        return {}

    # TODO(星图云接入点)：此处用 requests 调用 base_url，携带 token 与上述三个
    # 产品参数，解析返回的火烧云等级（如 优质大烧/大烧/中烧/小烧/微烧 或数值等级），
    # 映射到 0-100 分，再按 (date, window) 组织成：
    #   {(date, window): {"geovisearth": score, "grade": "...", "quality": "..."}}
    # 示例请求参数（以官方文档为准）：
    #   params = {"token": token, "productCode": product_code,
    #             "dataCode": data_code, "meteCode": mete_code,
    #             "start": start_date, "end": end_date}
    return {}



def run_crosscheck(cfg, days=None):
    """执行交叉验证，返回 {(date, window): info}。

    info 字段：scores(各模型评分) / mean / spread / confidence / agree。
    单个模型数据缺失时自动跳过；失败时返回空 dict（主预测不受影响）。
    """
    try:
        multi = weather.get_multi_model_forecast(cfg, days=days)
    except Exception:
        return {}

    daily_raw = multi.get("daily", {})
    per_window = {}
    for model in CROSSCHECK_MODELS:
        suffix = "_" + model
        hourly = _model_hourly(multi, model)
        if not _valid_hourly(hourly):
            continue
        daily = {"time": daily_raw.get("time", [])}
        for key in ("sunrise", "sunset"):
            full = key + suffix
            if full in daily_raw:
                daily[key] = daily_raw[full]
        feats = extract_day_features({"hourly": hourly, "daily": daily})
        for key, f in feats.items():
            score, _ = rule_score(f)
            per_window.setdefault(key, {})[model] = round(score, 1)

    result = {}
    for key, scores in per_window.items():
        if len(scores) < 2:
            continue
        vals = list(scores.values())
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        grades = {grade_of(v) for v in vals}
        result[key] = {
            "scores": scores,
            "mean": round(mean, 1),
            "spread": round(spread, 1),
            "confidence": confidence_of(spread),
            "agree": len(grades) == 1,
        }
    return result
