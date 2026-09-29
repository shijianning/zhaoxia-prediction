"""每日预测模块。

拉取未来预报 → 抽取窗口特征 → 规则评分 + 模型概率 → 融合评级。
"""
from . import crosscheck
from . import features as feat_mod
from . import logger as log_mod
from . import transect
from . import weather
from .glow_rules import chroma_index, grade_of, rule_score
from .model import GlowModel


def run_prediction(cfg):
    """执行预测，返回 (results, meta)。

    results: 按日期聚合的列表，每项含 morning/evening 两窗口的评分信息。
    meta: 预测时使用的模型信息。
    """
    log = log_mod.get_logger()

    data = weather.get_forecast(cfg)
    day_feats = feat_mod.extract_day_features(data)
    daily_wx = feat_mod.extract_daily_weather(data)

    # 合并空气质量（气溶胶 AOD）——用于"鲜艳度"判断（失败不影响主预测）
    try:
        aq = weather.get_air_quality(cfg)
        feat_mod.add_air_quality(day_feats, aq)
    except Exception as exc:
        log.warning("空气质量(AOD)获取失败，通透度因子将移出加权：%s", exc)

    # 加载模型；并施加真实标签闸门
    model = GlowModel.load(cfg["model"]["model_path"])
    min_real = int(cfg["model"].get("min_real_labels", 30))
    model_gated = False
    if model is not None and getattr(model, "real_labels", 0) < min_real:
        log.warning(
            "模型真实观测标注 %s 条 < 门槛 %s 条，本次不启用 ML，退回纯规则评分",
            getattr(model, "real_labels", 0), min_real,
        )
        model = None
        model_gated = True

    # 多模型交叉验证（失败不影响主预测）
    xcheck = crosscheck.run_crosscheck(cfg)

    # 第 4 方第三方对比：星图云官方火烧云预报（未配置 token 时为空，不参与）
    gvcheck = crosscheck.run_geovisearth_crosscheck(cfg)

    # 太阳方位剖面：沿日出/日落方位拉 0~500km 剖面，补上"方向性"评分。
    # 单次多坐标请求，失败即返回 {}，评分自动回落到原有 7 因子。
    tr_check = {}
    if transect.enabled(cfg, "single"):
        try:
            tr_check = transect.compute_transect(cfg, data)
        except Exception as exc:
            log.warning("太阳方位剖面取数失败，方向性因子移出加权：%s", exc)
            tr_check = {}
        if not tr_check:
            log.warning("太阳方位剖面为空（无云量数据或方位计算失败），方向性因子移出加权")

    for key, f in day_feats.items():
        tr = tr_check.get(key)
        if tr:
            f["transect"] = tr

        f["rule_score"], f["breakdown"] = rule_score(f)
        f["date"], f["window"] = key

        # 先挂交叉验证结果（供下方多源集成打分使用）
        xc = xcheck.get(key)
        if xc is not None:
            f["xcheck"] = xc

        # 基础分：有模型时规则分与模型概率各占一半，否则纯规则分
        if model is not None:
            f["prob"] = model.predict_proba(f)
            base = 0.5 * f["rule_score"] + 0.5 * f["prob"] * 100
        else:
            f["prob"] = None
            base = f["rule_score"]

        # 多源集成：若有三模型交叉验证均值，按一致性权重并入最终分
        # （融合"多源印证"思想：多个独立气象模式互相印证，结论一致时更可信）
        if xc is not None and xc.get("mean") is not None:
            f["final"] = round(0.7 * base + 0.3 * xc["mean"], 1)
            f["ensemble_base"] = round(base, 1)
        else:
            f["final"] = round(base, 1)

        f["grade"] = grade_of(f["final"])
        f["chroma"] = chroma_index(f["final"])

        # 挂上星图云第三方对比结果（若已配置）
        gvc = gvcheck.get(key)
        if gvc is not None:
            f["geovisearth"] = gvc

    # 按日期聚合
    by_date = {}
    for key, f in day_feats.items():
        date, window = key
        by_date.setdefault(date, {})[window] = f

    results = []
    for date in sorted(by_date.keys()):
        results.append({
            "date": date,
            "windows": by_date[date],
            "daily": daily_wx.get(date),
        })

    meta = {
        "city": cfg["city"]["name"],
        "has_model": model is not None,
        "model_type": cfg["model"]["type"] if model is not None else None,
        # 是否因"真实标注不足"被闸门拦下（报告里据此说明为何是纯规则模式）
        "model_gated": model_gated,
        "model_real_labels": getattr(model, "real_labels", None) if model is not None else None,
        "min_real_labels": min_real,
    }
    return results, meta
