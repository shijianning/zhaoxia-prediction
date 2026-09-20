"""多气象模型交叉验证模块。

用多个独立的全球气象模式（ECMWF IFS / NOAA GFS / DWD ICON）对同一地点、
同一日出/日落窗口各自计算规则评分，并给出模型间一致性/置信度，
用于佐证主预测：结论一致时可信度高，分歧大时说明该时段云况不稳定、预报不确定。

这是"用不同预测程序互相印证"的免费实现（无需 API Key）。
"""
import datetime as dt

import requests

from . import weather
from .features import extract_day_features
from .glow_rules import grade_of, rule_score, vividness_of

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
    """第 4 方第三方对比：星图云(GeoVisEarth)官方「火烧云点查询预报」API。

    接口：GET https://api.open.geovisearth.com/v2/grid/glow/day
    文档：https://open.geovisearth.com/support/document?docId=495
    返回指定经纬度未来 3 天（含当天）日出/日落时刻的火烧云质量与等级、
    蓝天指数与等级。这是对 SunsetBot / 莉景天气 这类「无公开 API」小程序的
    最佳官方替代数据源。

    未在 config 填入 token / enabled=false 时直接返回空 dict（不参与对比）。
    返回 {(date, window): {score, grade, quality, level}}，与 run_crosscheck 结构一致；
    score 已映射为 0-100 分，便于直接挂到报告上。
    """
    gv = (cfg.get("third_party") or {}).get("geovisearth") or {}
    if not gv.get("enabled") or not gv.get("token"):
        return {}

    token = gv["token"]
    base_url = gv.get(
        "base_url", "https://api.open.geovisearth.com/v2/grid/glow/day"
    )
    mete_codes = gv.get("mete_codes", "glow,aod")
    want_level = bool(gv.get("level", True))

    city = cfg["city"]
    days = days or cfg["forecast"]["days"]

    # 时间范围：今天 00 时 ~ (今天 + days) 23 时，格式 yyyyMMddHH
    today = dt.date.today()
    start_str = today.strftime("%Y%m%d") + "00"
    end_str = (today + dt.timedelta(days=days)).strftime("%Y%m%d") + "23"

    params = {
        "location": f"{city['longitude']},{city['latitude']}",  # 经度,纬度
        "start": start_str,
        "end": end_str,
        "meteCodes": mete_codes,
        "level": "true" if want_level else "false",
        "token": token,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return {}

    if payload.get("status") != 0:
        return {}

    result = payload.get("result") or {}
    mete_codes_list = result.get("meteCodes") or []
    datas = result.get("datas") or []

    # 找到 glow（火烧云质量）在 meteCodes 里的下标
    try:
        glow_idx = mete_codes_list.index("glow")
    except ValueError:
        return {}

    out = {}
    for item in datas:
        fc_time = item.get("fc_time") or ""
        if not fc_time:
            continue
        # fc_time 形如 "2026-09-20 06:30:00"（本地时区）
        date = fc_time[:10]
        hour = int(fc_time[11:13]) if len(fc_time) >= 13 else 12
        window = "morning" if hour < 12 else "evening"

        values = item.get("values") or []
        levels = item.get("levels") or []
        glow_val = values[glow_idx] if glow_idx < len(values) else None
        glow_lv = levels[glow_idx] if glow_idx < len(levels) else None

        score = _glow_to_score(glow_val, glow_lv)
        if score is None:
            continue
        out[(date, window)] = {
            "score": score,
            "grade": grade_of(score),
            "vivid": vividness_of(score),
            "quality": glow_val,
            "level": glow_lv,
        }
    return out


def _glow_to_score(value, level):
    """把星图云的火烧云质量(value)与等级(level)映射为 0-100 分。

    官方未公开确切的数值标定，这里采用经验映射（可随实测微调）：
    - value 为火烧云质量，示例中约 0-1；若返回 0-100 则直接使用。
    - level 为等级（示例 0-6），作为 value 缺失时的兜底。
    """
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = None
        if v is not None:
            if v <= 1.0:
                return round(v * 100, 1)
            return round(v, 1)

    if level is not None:
        try:
            lv = int(level)
        except (TypeError, ValueError):
            lv = None
        if lv is not None:
            return round(min(max(lv, 0), 6) / 6.0 * 100, 1)
    return None



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
