"""每日运行入口：训练(可选) + 预测 + 生成报告 + 控制台摘要。

用法：
    python daily_run.py               # 训练 + 预测 + 报告
    python daily_run.py --predict-only  # 仅预测，跳过训练
"""
import argparse
import datetime as dt
import sys

from src import predict, report, train
from src.config import load_config


def _print_summary(results, meta, train_result):
    print("=" * 56)
    print(f"  {meta['city']} 朝霞/晚霞预测")
    print("=" * 56)
    if train_result:
        s = train_result["stats"]
        if train_result.get("trained"):
            m = train_result["metrics"]
            real = s.get("posts", 0)
            if real == 0:
                print(f"[训练] 弱监督预训练 · 样本 {s['total']} (正 {s['positive']}/负 {s['negative']})"
                      f" · 暂无真实观测标注")
            else:
                print(f"[训练] 完成 · 样本 {s['total']} (真实观测 {real}/弱监督 {s['total'] - real})"
                      f" · 准确率 {m.get('accuracy', '—')}")
        else:
            print(f"[训练] 跳过 · {train_result.get('reason', '样本不足')}")
    print("-" * 56)
    for r in results:
        line_parts = []
        for w in ("morning", "evening"):
            f = r["windows"].get(w)
            if f is None:
                continue
            tag = "朝霞" if w == "morning" else "晚霞"
            line_parts.append(f"{tag} {f['final']:>5.0f}分({f['grade']})")
        print(f"  {r['date']}  " + "  |  ".join(line_parts))
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(description="朝霞晚霞每日预测")
    parser.add_argument("--predict-only", action="store_true",
                        help="跳过训练，仅执行预测")
    args = parser.parse_args()

    cfg = load_config()

    from src import logger as log_mod
    log = log_mod.setup_logging(cfg)
    log.info("===== daily_run 开始（predict_only=%s）=====", args.predict_only)

    try:
        _run(cfg, args)
        log.info("===== daily_run 完成 =====")
    except Exception as exc:
        log.exception("daily_run 运行失败")
        _alert_failure(cfg, exc)
        print(f"[错误] 运行失败：{exc}，详见 output/logs/run.log")
        return 1
    return 0


def _run(cfg, args):
    train_result = None
    if not args.predict_only:
        try:
            print("正在训练模型 ...")
            train_result = train.run_training(cfg)
        except Exception as exc:  # 训练失败不影响预测
            print(f"[警告] 训练失败：{exc}，继续预测 ...")

    print("正在拉取预报并预测 ...")
    results, meta = predict.run_prediction(cfg)

    report_path = report.save_report(cfg, results, meta, train_result)

    # 写入本地 MySQL（未启动则自动跳过，不影响报告生成）
    db_msg = _save_to_db(cfg, results, meta)

    # 微信推送（未配置则静默跳过）
    notify_msg = _send_notify(cfg, results, meta)

    _print_summary(results, meta, train_result)
    print(f"\n报告已生成：{report_path}")
    if db_msg:
        print(db_msg)
    if notify_msg:
        print(notify_msg)
    print("用浏览器打开该文件即可查看。")


def _alert_failure(cfg, exc):
    from src import notify
    try:
        msg = notify.send_failure_alert(cfg, exc)
        if msg:
            print(msg)
    except Exception:
        pass


def _save_to_db(cfg, results, meta):
    from src import database
    try:
        database.init_schema(cfg)
        database.sync_posts_from_csv(cfg)
        database.upsert_cities(cfg, [{
            "name": meta["city"],
            "lat": cfg["city"]["latitude"],
            "lon": cfg["city"]["longitude"],
            "major": False,
        }])
        ok = database.save_daily_results(cfg, results, meta["city"])
        return f"MySQL 已写入：{meta['city']} 预测历史已入库" if ok else "MySQL 未写入（可能未启动）"
    except Exception as exc:
        return f"MySQL 写入跳过：{exc}"


def _send_notify(cfg, results, meta):
    from src import notify
    try:
        return notify.send_notification(cfg, results, meta)
    except Exception as exc:
        return f"微信推送跳过：{exc}"


if __name__ == "__main__":
    sys.exit(main())
