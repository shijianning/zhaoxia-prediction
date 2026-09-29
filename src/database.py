"""MySQL 存储层。

把预测历史、城市列表、训练标注写入本地 MySQL（数据库名 zhaoxia）。

设计要点：
- 所有函数在 MySQL 未启动 / 未配置 / 连接失败时都优雅降级（返回 None/False/空），
  绝不影响主预测流程（预测仍会照常生成 HTML 报告）。
- **降级但不静默**：每个 except 都会经 `_log_warn()` 写一条 warning 到统一日志。
  定时任务没有可见终端，日志文件是排查问题的唯一依据；从前"无声失败"的写法
  会让 MySQL 挂掉表现为"分数莫名变低"。
- 表结构：
    cities        城市列表（name 唯一）
    predictions   每日朝霞/晚霞预测历史（city,date,window,source 唯一，幂等 upsert）
    posts         训练观测标注（帖子）

**单位约定（易踩坑）**：
- `visibility` 列存**米**（与 Open-Meteo 原始输出一致），报告展示时才除以 1000；
- `chroma` 为 0-10 的鲜艳度指数；`score`/`rule_score` 为 0-100 分；
- `spread` 为多模型交叉验证的极差（0-100 分）。
"""
import datetime as dt
import re

# SQL 标识符白名单：库名/列名虽来自本地配置，仍不应直接拼进 SQL
_IDENT_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def _log_warn(msg, exc=None):
    """降级时留痕。logger 未初始化时也不会抛异常。"""
    try:
        from .logger import get_logger
        log = get_logger()
        if exc is None:
            log.warning(msg)
        else:
            log.warning("%s: %s", msg, exc)
    except Exception:
        pass


def _safe_ident(name, what="标识符"):
    """校验 SQL 标识符（库名/表名/列名），非法则抛 ValueError。"""
    s = str(name)
    if not _IDENT_RE.match(s):
        raise ValueError(f"非法的 {what}: {s!r}（只允许字母/数字/下划线，长度 1-64）")
    return s


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
        chroma DOUBLE DEFAULT NULL,
        `spread` DOUBLE DEFAULT NULL,
        visibility DOUBLE DEFAULT NULL,   -- 单位：米
        sunrise VARCHAR(8) DEFAULT NULL,
        sunset VARCHAR(8) DEFAULT NULL,
        tmax DOUBLE DEFAULT NULL,
        tmin DOUBLE DEFAULT NULL,
        precip_sum DOUBLE DEFAULT NULL,
        precip_prob DOUBLE DEFAULT NULL,
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
    except Exception as exc:
        _log_warn("MySQL 探活失败", exc)
        return False


def init_schema(cfg):
    """创建数据库 + 三张表（幂等）。返回 True/False。"""
    d = db_conf(cfg)
    try:
        dbname = _safe_ident(d["database"], "数据库名")
    except ValueError as exc:
        _log_warn("数据库名非法，跳过初始化", exc)
        return False

    try:
        # 先连到无库连接，建库
        conn = connect_server(cfg)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `%s` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci" % dbname
            )
        conn.close()
    except Exception as exc:
        _log_warn("建库失败（MySQL 未启动或权限不足），跳过入库", exc)
        return False

    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt)
        conn.close()
        # 老库升级：为 predictions 表补上后来新增的列（幂等）
        _migrate_predictions(cfg)
        return True
    except Exception as exc:
        _log_warn("建表失败", exc)
        return False


# 后来新增的列（老库缺列时自动 ALTER 补齐，不删旧数据）
_PREDICTIONS_NEW_COLUMNS = [
    ("chroma", "DOUBLE DEFAULT NULL"),
    ("spread", "DOUBLE DEFAULT NULL"),
    ("visibility", "DOUBLE DEFAULT NULL"),
    ("sunrise", "VARCHAR(8) DEFAULT NULL"),
    ("sunset", "VARCHAR(8) DEFAULT NULL"),
    ("tmax", "DOUBLE DEFAULT NULL"),
    ("tmin", "DOUBLE DEFAULT NULL"),
    ("precip_sum", "DOUBLE DEFAULT NULL"),
    ("precip_prob", "DOUBLE DEFAULT NULL"),
]


def _migrate_predictions(cfg):
    """把 predictions 表缺的列补上（MySQL 不支持 ADD COLUMN IF NOT EXISTS，需查列名）。"""
    try:
        conn = connect(cfg)
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM predictions")
            existing = {row[0].lower() for row in cur.fetchall()}
            for col, coldef in _PREDICTIONS_NEW_COLUMNS:
                if col not in existing:
                    cur.execute(
                        "ALTER TABLE predictions ADD COLUMN `%s` %s"
                        % (_safe_ident(col, "列名"), coldef)
                    )
        conn.close()
        return True
    except Exception as exc:
        _log_warn("predictions 表结构升级失败", exc)
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
    except Exception as exc:
        _log_warn("写入城市列表失败", exc)
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
    except Exception as exc:
        _log_warn("读取城市列表失败", exc)
        return []


# ---------------------------------------------------------------------------
# 预测历史
# ---------------------------------------------------------------------------
def upsert_predictions(cfg, rows):
    """批量写入预测记录（city,date,window,source 唯一，重复则覆盖）。

    rows: 列表，每项 dict，字段：
        city / date / window / score（必填）
        vivid / grade / rule_score / model_prob / aod / pm2_5 / source
        chroma / spread / visibility / sunrise / sunset / tmax / tmin /
        precip_sum / precip_prob（可选，缺失写 NULL）
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
                    " model_prob, aod, pm2_5, chroma, `spread`, visibility, "
                    " sunrise, sunset, tmax, tmin, precip_sum, precip_prob, source) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    " %s, %s, %s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE "
                    "score=VALUES(score), vivid=VALUES(vivid), grade=VALUES(grade), "
                    "rule_score=VALUES(rule_score), model_prob=VALUES(model_prob), "
                    "aod=VALUES(aod), pm2_5=VALUES(pm2_5), chroma=VALUES(chroma), "
                    "`spread`=VALUES(`spread`), visibility=VALUES(visibility), "
                    "sunrise=VALUES(sunrise), sunset=VALUES(sunset), "
                    "tmax=VALUES(tmax), tmin=VALUES(tmin), "
                    "precip_sum=VALUES(precip_sum), precip_prob=VALUES(precip_prob), "
                    "source=VALUES(source)",
                    (
                        r["city"], r["date"], r["window"], r["score"],
                        r.get("vivid"), r.get("grade"), r.get("rule_score"),
                        r.get("model_prob"), r.get("aod"), r.get("pm2_5"),
                        r.get("chroma"), r.get("spread"), r.get("visibility"),
                        r.get("sunrise"), r.get("sunset"), r.get("tmax"),
                        r.get("tmin"), r.get("precip_sum"), r.get("precip_prob"),
                        r.get("source", "daily"),
                    ),
                )
        conn.close()
        return True
    except Exception as exc:
        _log_warn("写入预测历史失败", exc)
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
    except Exception as exc:
        _log_warn("统计预测条数失败", exc)
        return None


# ---------------------------------------------------------------------------
# 训练标注（帖子）
# ---------------------------------------------------------------------------
def sync_posts_from_csv(cfg):
    """把 data/raw/posts.csv 同步进 posts 表（追加，不删旧）。"""
    import csv
    import os
    csv_path = cfg["data"]["posts_csv"]
    if not os.path.exists(csv_path):
        return False
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("date") or not r.get("window"):
                continue
            try:
                glow = int(r.get("glow", 0))
            except (TypeError, ValueError):
                continue
            rows.append({
                "date": r["date"], "window": r["window"],
                "glow": glow,
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
    except Exception as exc:
        _log_warn("写入训练标注失败", exc)
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
    except Exception as exc:
        _log_warn("读取训练标注失败", exc)
        return []


# ---------------------------------------------------------------------------
# 高层适配：把预测结果转成 rows 并入库
# ---------------------------------------------------------------------------
def save_national_results(cfg, results, source="national"):
    """把全国地图预测结果写入 predictions 表。

    results: {city: {date: {window: {score, vivid, grade}, daily: {...}}}}
    只写入 morning/evening 两个窗口，跳过 daily 天气概览字段。
    """
    rows = []
    for city, days in results.items():
        for date, wins in days.items():
            for window in ("morning", "evening"):
                info = wins.get(window)
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
             [{date, windows: {window: {final, grade, rule_score, prob, aod, ...}},
               daily: {sunrise, sunset, tmax, tmin, precip_sum, precip_prob}}]
    """
    from .glow_rules import vividness_of

    def _hhmm(iso):
        """'2026-09-20T06:32' -> '06:32'（本地时区，Open-Meteo 已按 timezone 返回）。"""
        if not iso:
            return None
        return str(iso)[11:16]

    rows = []
    for r in results:
        daily = r.get("daily") or {}
        for window, f in (r.get("windows") or {}).items():
            score = f.get("final")
            xc = f.get("xcheck") or {}
            rows.append({
                "city": city, "date": r["date"], "window": window,
                "score": score,
                "vivid": vividness_of(score) if score is not None else None,
                "grade": f.get("grade"),
                "rule_score": f.get("rule_score"),
                "model_prob": f.get("prob"),
                "aod": f.get("aod"),
                "pm2_5": f.get("pm2_5"),
                "chroma": f.get("chroma"),
                "spread": xc.get("spread"),
                "visibility": f.get("visibility"),
                "sunrise": _hhmm(daily.get("sunrise")),
                "sunset": _hhmm(daily.get("sunset")),
                "tmax": daily.get("tmax"),
                "tmin": daily.get("tmin"),
                "precip_sum": daily.get("precip_sum"),
                "precip_prob": daily.get("precip_prob"),
                "source": "daily",
            })
    return upsert_predictions(cfg, rows)
