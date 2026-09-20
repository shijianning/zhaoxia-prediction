"""回测与阈值校准脚本。

用 data/raw/posts.csv（或 MySQL 镜像）里的历史观测标签，评估当前
「规则分 + 模型概率」的预测准确率，并对推送阈值做扫描校准，形成
「预测 -> 观测 -> 回测 -> 再校准」闭环。

用法：
    python backtest.py                 # 用默认阈值扫描
    python backtest.py --thresholds 60,65,70,75,80
"""
import argparse
import datetime as dt
import sys

from src import features as feat_mod
from src import posts as posts_mod
from src import train
from src import weather
from src.config import load_config
from src.glow_rules import chroma_index, grade_of, rule_score, vividness_of
from src.model import GlowModel

DEFAULT_THRESHOLDS = [45, 50, 55, 60, 65, 70, 75, 80]


def _load_features(cfg, dates):
    """按标签日期范围批量拉历史气象 + 空气质量，返回 {(date, window): feature}。"""
    if not dates:
        return {}
    start = min(dates)
    end = max(dates)
    data = weather.get_historical(cfg, start, end)
    day_feats = feat_mod.extract_day_features(data)
    try:
        aq = weather.get_historical_air_quality(cfg, start, end)
        feat_mod.add_air_quality(day_feats, aq)
    except Exception:
        pass
    for key, f in day_feats.items():
        f["rule_score"], f["breakdown"] = rule_score(f)
        f["date"], f["window"] = key
    return day_feats


def _final_score(f, model):
    """复现 predict.py 的最终分（回测不含多模型交叉验证）。"""
    if model is not None:
        prob = model.predict_proba(f)
        return round(0.5 * f["rule_score"] + 0.5 * prob * 100, 1)
    return round(f["rule_score"], 1)


def main():
    parser = argparse.ArgumentParser(description="朝霞晚霞预测回测校准")
    parser.add_argument("--thresholds", type=str, default="",
                        help="逗号分隔的阈值列表，如 60,65,70,75")
    args = parser.parse_args()

    cfg = load_config()
    thresholds = (
        [int(x) for x in args.thresholds.split(",") if x.strip()]
        if args.thresholds.strip()
        else DEFAULT_THRESHOLDS
    )

    # 1) 标签（统一真源）
    labels = train.load_labels(cfg)
    if not labels:
        print("没有观测标注（data/raw/posts.csv 为空），无法回测。")
        print("请先补充真实观测：date,window,glow(0/1),source,note")
        return 1

    # 2) 特征
    dates = {p["date"] for p in labels}
    day_feats = _load_features(cfg, dates)

    model = GlowModel.load(cfg["model"]["model_path"])

    # 3) 逐标签计算最终分
    print("=" * 64)
    print("  回测明细（实际出霞 1 / 未出霞 0）")
    print("=" * 64)
    print(f"{'日期':<12}{'时段':<6}{'实际':<4}{'最终分':<7}{'评级':<6}{'鲜艳度':<8}")
    print("-" * 64)
    rows = []
    n_miss_feat = 0
    for p in labels:
        key = (p["date"], p["window"])
        f = day_feats.get(key)
        if f is None:
            n_miss_feat += 1
            continue
        score = _final_score(f, model)
        tag = "朝霞" if p["window"] == "morning" else "晚霞"
        rows.append({"glow": p["glow"], "score": score})
        print(f"{p['date']:<12}{tag:<6}{p['glow']:<4}{score:<7.0f}"
              f"{grade_of(score):<6}{vividness_of(score):<6}"
              f"🔥{chroma_index(score)}")
    if n_miss_feat:
        print(f"（{n_miss_feat} 条标注因历史数据缺失未纳入）")
    print("=" * 64)

    if not rows:
        print("没有可回测的样本（历史特征缺失）。")
        return 1

    # 4) 阈值扫描
    n = len(rows)
    print(f"\n样本总数：{n}（正例 {sum(1 for r in rows if r['glow'] == 1)} / "
          f"负例 {sum(1 for r in rows if r['glow'] == 0)}）")
    print("-" * 64)
    print(f"{'阈值':<6}{'准确率':<8}{'精确率':<8}{'召回率':<8}{'F1':<7}{'漏报':<6}")
    print("-" * 64)
    best_thr, best_f1 = None, -1
    for t in thresholds:
        tp = fp = tn = fn = 0
        for r in rows:
            pred = 1 if r["score"] >= t else 0
            if pred == 1 and r["glow"] == 1:
                tp += 1
            elif pred == 1 and r["glow"] == 0:
                fp += 1
            elif pred == 0 and r["glow"] == 0:
                tn += 1
            else:
                fn += 1
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f"{t:<6}{acc:<8.2%}{prec:<8.2%}{rec:<8.2%}{f1:<7.3f}{fn:<6}")
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    print("-" * 64)
    print(f"\n建议推送阈值：{best_thr}（F1 最高 {best_f1:.3f}）")
    print("提示：F1 兼顾「不错过好霞」和「少打扰」；若更怕漏掉好霞可降低阈值，")
    print("      若更怕被频繁打扰可提高阈值。回测样本越多，结论越可靠。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
