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
    args = parser.parse_args()

    cfg = load_config()
    cities = national_map.load_cities()
    if args.cities and args.cities > 0:
        cities = cities[: args.cities]

    if args.render_only:
        results, failures = national_map.load_cache(cfg)
        if results is None:
            print("没有缓存，请先完整运行一次 `python national_map.py`。")
            return 1
        path = national_map.save_national_map(cfg, cities, results, failures, days=args.days)
        print(f"已从缓存重新渲染：{path}")
        return 0

    print(f"开始预测 {len(cities)} 个城市的朝霞/晚霞（并发 {args.workers}）...")
    results, failures = national_map.run_national(
        cfg, cities, days=args.days, workers=args.workers, progress=_progress
    )

    national_map.save_cache(cfg, results, failures)
    path = national_map.save_national_map(cfg, cities, results, failures, days=args.days)

    # 写入本地 MySQL（未启动则自动跳过）
    db_note = _save_to_db(cfg, results)

    print("-" * 56)
    print(f"成功 {len(results)} 城，失败 {len(failures)} 城")
    if failures:
        for name, msg in list(failures.items())[:10]:
            print(f"  ✗ {name}: {msg}")
    print(db_note)
    print(f"\n地图已生成：{path}")
    print("用浏览器打开该文件即可查看全国朝霞晚霞预测地图。")


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
