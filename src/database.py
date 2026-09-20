"""MySQL 存储层。

把预测历史、城市列表、训练标注写入本地 MySQL（数据库名 zhaoxia）。

设计要点：
- 所有函数在 MySQL 未启动 / 未配置 / 连接失败时都优雅降级（返回 None/False/空），
  绝不影响主预测流程（预测仍会照常生成 HTML 报告）。
- 表结构：
    cities        城市列表（name 唯一）
    predictions   每日朝霞/晚霞预测历史（city,date,window 唯一，幂等 upsert）
    posts         训练观测标注（帖子）
"""
import datetime as dt

DEFAULT_DB = {
    "enabled": True,
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "zhaoxia",
}

# 建表语句（幂等）
_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS cities (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) NOT NULL UNIQUE,
        lat DOUBLE NOT NULL,
        lon DOUBLE NOT NULL,
        major TINYINT(1) NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        city VARCHAR(50) NOT NULL,
        `date` DATE NOT NULL,
        `window` VARCHAR(10) NOT NULL,
        score DOUBLE NOT NULL,
        vivid VARCHAR(20) DEFAULT NULL,
        grade VARCHAR(20) DEFAULT NULL,
        rule_score DOUBLE DEFAULT NULL,
        model_prob DOUBLE DEFAULT NULL,
        aod DOUBLE DEFAULT NULL,
        pm2_5 DOUBLE DEFAULT NULL,
        source VARCHAR(20) DEFAULT 'daily',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_city_date_window_src (city, `date`, `window`, source),
        KEY idx_date (`date`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
        id INT AUTO_INCREMENT PRIMARY KEY,
        `date` DATE NOT NULL,
        `window` VARCHAR(10) NOT NULL,
        glow TINYINT(1) NOT NULL,
        source VARCHAR(50) DEFAULT NULL,
        note TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_date_window (`date`, `window`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


def db_conf(cfg):
    """合并默认值与 config.yaml 的 database 段。"""
    d = dict(DEFAULT_DB)
    d.update(cfg.get("database") or {})
    return d


def connect(cfg):
    """连接指定数据库。失败抛异常。"""
    import pymysql
    d = db_conf(cfg)
    return pymysql.connect(
        host=d["host"], port=int(d["port"]), user=d["user"],
        password=d["password"], database=d["database"],
        charset="utf8mb4", connect_timeout=5, autocommit=True,
    )


def connect_server(cfg):
    """连接 MySQL 服务器本身（不指定库，用于建库/探活）。失败抛异常。"""
    import pymysql
    d = db_conf(cfg)
    return pymysql.connect(
        host=d["host"], port=int(d["port"]), user=d["user"],
        password=d["password"], charset="utf8mb4",
        connect_timeout=5, autocommit=True,
    )


def is_available(cfg):
    """探测 MySQL 服务器是否可用（不看具体库是否存在）。"""
    if not (cfg.get("database") or {}).get("enabled", True):
        return False
    try:
        conn = connect_server(cfg)
        conn.close()
        return True
    except Exception:
        return False


def init_schema(cfg):
    """创建数据库 + 三张表（幂等）。返回 True/False。"""
    d = db_conf(cfg)
    try:
        # 先连到无库连接，建库
        conn = connect_server(cfg)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `%s` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci" % d["database"]
            )
        conn.close()
    except Exception:
        return False

    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt)
        conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 城市
# ---------------------------------------------------------------------------
def upsert_cities(cfg, cities):
    """写入城市列表（name 唯一，重复则更新坐标）。"""
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for c in cities:
                cur.execute(
                    "INSERT INTO cities (name, lat, lon, major) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE lat=VALUES(lat), lon=VALUES(lon), major=VALUES(major)",
                    (c["name"], c["lat"], c["lon"], 1 if c.get("major") else 0),
                )
        conn.close()
        return True
    except Exception:
        return False


def load_cities(cfg):
    """读取城市列表，返回 [{name, lat, lon, major}]。"""
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            cur.execute("SELECT name, lat, lon, major FROM cities ORDER BY id")
            rows = cur.fetchall()
        conn.close()
        return [{"name": r[0], "lat": r[1], "lon": r[2], "major": bool(r[3])} for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 预测历史
# ---------------------------------------------------------------------------
def upsert_predictions(cfg, rows):
    """批量写入预测记录（city,date,window 唯一，重复则覆盖）。

    rows: 列表，每项 dict，字段：
        city / date / window / score（必填）
        vivid / grade / rule_score / model_prob / aod / pm2_5 / source（可选）
    """
    if not rows:
        return False
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for r in rows:
                cur.execute(
                    "INSERT INTO predictions "
                    "(city, `date`, `window`, score, vivid, grade, rule_score, "
                    " model_prob, aod, pm2_5, source) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "score=VALUES(score), vivid=VALUES(vivid), grade=VALUES(grade), "
                    "rule_score=VALUES(rule_score), model_prob=VALUES(model_prob), "
                    "aod=VALUES(aod), pm2_5=VALUES(pm2_5), source=VALUES(source)",
                    (
                        r["city"], r["date"], r["window"], r["score"],
                        r.get("vivid"), r.get("grade"), r.get("rule_score"),
                        r.get("model_prob"), r.get("aod"), r.get("pm2_5"),
                        r.get("source", "daily"),
                    ),
                )
        conn.close()
        return True
    except Exception:
        return False


def count_predictions(cfg):
    """统计预测记录条数。"""
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM predictions")
            n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 训练标注（帖子）
# ---------------------------------------------------------------------------
def sync_posts_from_csv(cfg):
    """把 data/raw/posts.csv 同步进 posts 表（追加，不删旧）。"""
    import csv
    import os
    try:
        from .config import BASE_DIR
    except Exception:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = cfg["data"]["posts_csv"]
    if not os.path.exists(csv_path):
        return False
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("date") or not r.get("window"):
                continue
            rows.append({
                "date": r["date"], "window": r["window"],
                "glow": int(r.get("glow", 0)),
                "source": r.get("source") or "manual",
                "note": r.get("note"),
            })
    return insert_posts(cfg, rows)


def insert_posts(cfg, posts):
    """写入训练标注。返回 True/False。"""
    if not posts:
        return False
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for p in posts:
                cur.execute(
                    "INSERT INTO posts (`date`, `window`, glow, source, note) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "glow=VALUES(glow), source=VALUES(source), note=VALUES(note)",
                    (p["date"], p["window"], int(p["glow"]),
                     p.get("source"), p.get("note")),
                )
        conn.close()
        return True
    except Exception:
        return False


def load_posts(cfg):
    """读取全部训练标注，返回 [{date, window, glow, source, note}]。"""
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            cur.execute("SELECT `date`, `window`, glow, source, note FROM posts ORDER BY `date`")
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "date": r[0].isoformat() if isinstance(r[0], dt.date) else str(r[0]),
                "window": r[1], "glow": int(r[2]),
                "source": r[3], "note": r[4],
            }
            for r in rows
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 高层适配：把预测结果转成 rows 并入库
# ---------------------------------------------------------------------------
def save_national_results(cfg, results, source="national"):
    """把全国地图预测结果写入 predictions 表。

    results: {city: {date: {window: {score, vivid, grade}}}}
    """
    rows = []
    for city, days in results.items():
        for date, wins in days.items():
            for window, info in wins.items():
                if not info:
                    continue
                rows.append({
                    "city": city, "date": date, "window": window,
                    "score": info.get("score"),
                    "vivid": info.get("vivid"),
                    "grade": info.get("grade"),
                    "rule_score": info.get("score"),
                    "source": source,
                })
    return upsert_predictions(cfg, rows)


def save_daily_results(cfg, results, city):
    """把单城每日报告结果写入 predictions 表。

    results: predict.run_prediction 返回的 results 列表
             [{date, windows: {window: {final, grade, rule_score, prob, aod, ...}}}]
    """
    from .glow_rules import vividness_of
    rows = []
    for r in results:
        for window, f in (r.get("windows") or {}).items():
            score = f.get("final")
            rows.append({
                "city": city, "date": r["date"], "window": window,
                "score": score,
                "vivid": vividness_of(score) if score is not None else None,
                "grade": f.get("grade"),
                "rule_score": f.get("rule_score"),
                "model_prob": f.get("prob"),
                "aod": f.get("aod"),
                "pm2_5": f.get("pm2_5"),
                "source": "daily",
            })
    return upsert_predictions(cfg, rows)
