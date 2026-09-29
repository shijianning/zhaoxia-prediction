"""全国朝霞/晚霞预测地图 —— 入口。

用法：
    python national_map.py               # 预测全国城市并生成 output/national_map.html
    python national_map.py --cities 10   # 只跑前 10 个城市（调试用）
    python national_map.py --workers 8   # 指定并发数
"""
import argparse
import sys

from src import national_map
from src.config import load_config


def _progress(done, total, name, tag=""):
    print(f"  [{done}/{total}]{tag} {name} 完成")


def main():
    parser = argparse.ArgumentParser(description="全国朝霞晚霞预测地图")
    parser.add_argument("--cities", type=int, default=0,
                        help="只跑前 N 个城市（0 表示全部）")
    parser.add_argument("--workers", type=int, default=6,
                        help="并发请求数")
    parser.add_argument("--days", type=int, default=2,
                        help="预测未来几天")
    parser.add_argument("--render-only", action="store_true",
                        help="从缓存重新渲染地图，不重新拉取数据")
    parser.add_argument("--resume", action="store_true",
                        help="跳过今天缓存里已成功的城市，只跑剩余城市（用于分批补齐）")
    args = parser.parse_args()

    cfg = load_config()

    from src import logger as log_mod
    log = log_mod.setup_logging(cfg)
    log.info("===== national_map 开始（cities=%s, workers=%s, days=%s）=====",
             args.cities or "全部", args.workers, args.days)

    try:
        _run(cfg, args)
        log.info("===== national_map 完成 =====")
        return 0
    except Exception as exc:
        log.exception("national_map 运行失败")
        _alert_failure(cfg, exc)
        print(f"[错误] 运行失败：{exc}，详见 output/logs/run.log")
        return 1


def _run(cfg, args):
    cities = national_map.load_cities()
    if args.cities and args.cities > 0:
        cities = cities[: args.cities]

    if args.render_only:
        results, failures = national_map.load_cache(cfg)
        if results is None:
            print("没有缓存，请先完整运行一次 `python national_map.py`。")
            return
        path = national_map.save_national_map(cfg, cities, results, failures or {}, days=args.days)
        print(f"已从缓存重新渲染：{path}")
        return

    todo = cities
    if args.resume:
        have = national_map.cached_city_names(cfg)
        todo = [c for c in cities if c["name"] not in have]
        print(f"--resume：今天缓存里已有 {len(have)} 城，本次只跑剩余 {len(todo)} 城。")
        if not todo:
            results, failures = national_map.load_cache(cfg)
            path = national_map.save_national_map(cfg, cities, results or {}, failures or {},
                                                  days=args.days)
            print(f"全部城市均已有缓存，直接重绘：{path}")
            return

    print(f"开始预测 {len(todo)} 个城市的朝霞/晚霞（并发 {args.workers}）...")
    results, failures = national_map.run_national(
        cfg, todo, days=args.days, workers=args.workers, progress=_progress
    )

    # 与今天的旧缓存合并：若本轮被限流熔断提前收尾，可稍后用 --resume 补齐
    national_map.save_cache(cfg, results, failures)
    all_results, all_failures = national_map.load_cache(cfg)
    all_results = all_results or {}
    all_failures = all_failures or {}

    path = national_map.save_national_map(cfg, cities, all_results, all_failures, days=args.days)

    # 写入本地 MySQL（未启动则自动跳过）
    db_note = _save_to_db(cfg, all_results)

    print("-" * 56)
    print(f"本轮成功 {len(results)} 城；合并缓存后共 {len(all_results)} 城有数据，"
          f"{len(all_failures)} 城仍缺")
    if all_failures:
        for name, msg in list(all_failures.items())[:10]:
            print(f"  ✗ {name}: {msg}")
        if len(all_failures) > 10:
            print(f"  ... 另有 {len(all_failures) - 10} 城")
        print("  提示：等 Open-Meteo 配额恢复后运行 "
              "`python national_map.py --resume` 只补剩余城市。")
    print(db_note)
    print(f"\n地图已生成：{path}")
    print("用浏览器打开该文件即可查看全国朝霞晚霞预测地图。")


def _alert_failure(cfg, exc):
    from src import notify
    try:
        msg = notify.send_failure_alert(cfg, exc)
        if msg:
            print(msg)
    except Exception:
        pass


def _save_to_db(cfg, results):
    from src import database
    try:
        database.init_schema(cfg)
        ok = database.save_national_results(cfg, results, source="national")
        n = database.count_predictions(cfg)
        return f"MySQL 已写入：本次 {len(results)} 城预测入库（predictions 表累计 {n} 条）" if ok else "MySQL 未写入（可能未启动，数据仍已存缓存文件）"
    except Exception as exc:
        return f"MySQL 写入跳过：{exc}"


if __name__ == "__main__":
    sys.exit(main())
