"""每日训练模块。

流程：
1. 拉取历史逐小时气象数据（默认 120 天，走 Historical Forecast API，
   字段与推理侧的预报逐字段一致，避免 train-serve skew）；
2. 在每个日出/日落窗口抽取特征 + 规则评分；
3. 组装训练标签：
   a. 帖子观测标注（posts.csv，正负例，最可靠）
   b. 坏天气自动负样本（明显雨/阴 → 0）
   c. 弱监督 bootstrap（规则分高 → 1，低 → 0，中段丢弃）
4. **真实标签闸门**：真实观测标注不足 `model.min_real_labels`（默认 30）时，
   一律不训练不保存，直接退回纯规则模式 —— 因为 (c) 类标签由规则分生成，
   用它训练出的指标衡量的是"能否复刻规则引擎"，没有预测意义；
5. 真实标注充足时训练并保存模型，评估采用**时序留出且留出集只含真实观测**。
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


def load_labels(cfg):
    """统一训练标签真源，避免 CSV 与 MySQL 双源漂移。

    约定：`data/raw/posts.csv` 为标签真源（可进 git、可手动编辑）；
    MySQL `posts` 表仅作镜像。加载流程：
      1. 读 CSV（真源）；
      2. 若 MySQL 可用，先把 CSV 同步进表（CSV -> 表）；
      3. 再读表，把「表里有、CSV 没有」的标注合并进来（表 -> 内存），
         保证两端都不丢数据。MySQL 不可用时静默回退到仅 CSV。
    """
    posts = posts_mod.load_posts(cfg["data"]["posts_csv"])
    seen = {(p["date"], p["window"]) for p in posts}
    try:
        from . import database
        database.sync_posts_from_csv(cfg)  # CSV -> 表（真源优先）
        for p in database.load_posts(cfg):
            key = (p["date"], p["window"])
            if key not in seen:
                posts.append(p)
                seen.add(key)
    except Exception:
        pass
    return posts


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

    # 帖子标注（统一真源：CSV 为准 + MySQL 镜像合并）
    posts = load_labels(cfg)
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
        # 能匹配到特征、真正进入训练集的真实观测行数（留出集也是从这些里取）
        "post_rows": sum(1 for r in rows if r.get("source") == "post"),
        "auto_negative": n_auto_neg,
        "weak_positive": n_weak_pos,
        "weak_negative": n_weak_neg,
        "positive": sum(1 for r in rows if r["glow"] == 1),
        "negative": sum(1 for r in rows if r["glow"] == 0),
    }
    return rows, stats


def run_training(cfg):
    """执行训练，返回结果字典。

    **真实标签闸门**：真实观测标注少于 `model.min_real_labels` 时一律不训练、
    不保存模型。原因是弱监督标签由 `rule_score` 生成，而 `rule_score` 又是
    （或曾是与）模型输入的确定性函数，用它训练出的模型指标衡量的是
    "能否复刻规则引擎"，不能用来宣称预测能力。宁可退回纯规则评分，
    也不给出一个统计上站不住的"模型置信度"。
    """
    rows, stats = build_training_rows(cfg)

    n_pos = stats["positive"]
    n_neg = stats["negative"]
    n_real = stats["post_rows"]
    min_samples = cfg["model"]["min_samples"]
    min_real = int(cfg["model"].get("min_real_labels", 30))
    holdout_ratio = float(cfg["model"].get("holdout_ratio", 0.3))

    if n_real < min_real:
        return {
            "trained": False,
            "stats": stats,
            "metrics": {},
            "reason": (
                f"真实观测标注 {n_real} 条 < 门槛 {min_real} 条，暂用纯规则评分"
                f"（弱监督标签由规则分生成，不足以支撑模型评估，"
                f"请补充 data/raw/posts.csv 的真实观测）"
            ),
        }

    if len(rows) < min_samples or n_pos < 5 or n_neg < 5:
        return {"trained": False, "stats": stats, "metrics": {},
                "reason": "训练样本不足，暂用纯规则评分"}

    model = GlowModel(cfg["model"]["type"])
    metrics = model.train(rows, holdout_ratio=holdout_ratio, real_labels=stats["posts"])
    if not metrics.get("ok"):
        return {"trained": False, "stats": stats, "metrics": {},
                "reason": metrics.get("reason", "训练未成功")}

    model.save(cfg["model"]["model_path"])
    return {"trained": True, "stats": stats, "metrics": metrics}
