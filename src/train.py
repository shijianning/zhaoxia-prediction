"""每日训练模块。

流程：
1. 拉取历史逐小时气象数据（默认 120 天）；
2. 在每个日出/日落窗口抽取特征 + 规则评分；
3. 组装训练标签：
   a. 帖子观测标注（posts.csv，正负例，最可靠）
   b. 坏天气自动负样本（明显雨/阴 → 0）
   c. 弱监督 bootstrap（规则分高 → 1，低 → 0，中段丢弃）
4. 样本足够则训练模型并保存；否则退回纯规则模式。
"""
import datetime as dt

from . import features as feat_mod
from . import posts as posts_mod
from . import weather
from .glow_rules import rule_score
from .model import GlowModel


def _is_clearly_bad(f):
    """明显不适合出霞的坏天气：有雨，或全阴。"""
    return f["precip"] >= 0.5 or f["cloud_cover"] >= 95


def build_training_rows(cfg):
    """组装训练样本（特征 + glow 标签 + 来源）。"""
    history_days = cfg["data"]["history_days"]
    end = dt.date.today()
    start = end - dt.timedelta(days=history_days)

    data = weather.get_historical(cfg, start.isoformat(), end.isoformat())
    day_feats = feat_mod.extract_day_features(data)

    # 合并历史空气质量（气溶胶 AOD）；失败则无 aod，规则分给中性通透度
    try:
        aq = weather.get_historical_air_quality(cfg, start.isoformat(), end.isoformat())
        feat_mod.add_air_quality(day_feats, aq)
    except Exception:
        pass

    # 给每个窗口补上规则分与日期/时段
    for key, f in day_feats.items():
        f["rule_score"], _ = rule_score(f)
        f["date"], f["window"] = key

    # 帖子标注
    posts = posts_mod.load_posts(cfg["data"]["posts_csv"])
    post_map = {(p["date"], p["window"]): p["glow"] for p in posts}

    rows = []
    n_auto_neg = 0
    n_weak_pos = 0
    n_weak_neg = 0
    for key, f in day_feats.items():
        row = dict(f)
        if key in post_map:
            row["glow"] = post_map[key]
            row["source"] = "post"
        elif _is_clearly_bad(f):
            row["glow"] = 0
            row["source"] = "auto-negative"
            n_auto_neg += 1
        elif cfg["model"]["bootstrap"]:
            s = f["rule_score"]
            if s >= 65:
                row["glow"] = 1
                row["source"] = "weak-positive"
                n_weak_pos += 1
            elif s <= 35:
                row["glow"] = 0
                row["source"] = "weak-negative"
                n_weak_neg += 1
            else:
                continue  # 中间地带，标签不可靠，跳过
        else:
            continue
        rows.append(row)

    stats = {
        "total": len(rows),
        "posts": len(posts),
        "auto_negative": n_auto_neg,
        "weak_positive": n_weak_pos,
        "weak_negative": n_weak_neg,
        "positive": sum(1 for r in rows if r["glow"] == 1),
        "negative": sum(1 for r in rows if r["glow"] == 0),
    }
    return rows, stats


def run_training(cfg):
    """执行训练，返回结果字典。"""
    rows, stats = build_training_rows(cfg)

    n_pos = stats["positive"]
    n_neg = stats["negative"]
    min_samples = cfg["model"]["min_samples"]

    if len(rows) >= min_samples and n_pos >= 5 and n_neg >= 5:
        model = GlowModel(cfg["model"]["type"])
        metrics = model.train(rows)
        model.save(cfg["model"]["model_path"])
        return {"trained": True, "stats": stats, "metrics": metrics}
    return {"trained": False, "stats": stats,
            "metrics": {}, "reason": "训练样本不足，暂用纯规则评分"}
