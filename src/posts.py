"""帖子/观测数据接入模块。

训练标签的来源：
1. 手动/导出的观测标注 CSV（最可靠）
   列：date, window, glow, source, note
   - date: 'YYYY-MM-DD'
   - window: 'morning'(朝霞) / 'evening'(晚霞)
   - glow: 1(看到了好霞) / 0(没看到/平淡)
   - source: 数据来源(如 weibo/xiaohongshu/manual)
   - note: 备注

2. 文本关键词推断：从爬取的帖子正文自动推断是否出霞（best-effort）。

3. 自动负样本：由 train.py 依据坏天气自动生成（见 train.py）。
"""
import csv
import os


# 出霞关键词 / 未出霞关键词
GLOW_KEYWORDS = ["朝霞", "晚霞", "火烧云", "霞", "染红", "晚霞刷屏",
                 "日出", "日落", "橘红", "橙红", "紫霞", "天边红"]
NEG_KEYWORDS = ["阴天", "下雨", "雾霾", "灰蒙蒙", "没看到", "没等到",
                "白跑", "平淡", "没有霞"]


def load_posts(path):
    """读取观测标注 CSV，返回字典列表。"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "date": r["date"].strip(),
                    "window": r["window"].strip(),
                    "glow": int(r["glow"]),
                    "source": r.get("source", "").strip(),
                    "note": r.get("note", "").strip(),
                })
            except (KeyError, ValueError):
                continue
    return rows


def infer_label_from_text(text, window=None):
    """从帖子文本推断 (window, glow)。

    返回 (window, glow) 或 None（无法判断时）。
    """
    if not text:
        return None
    if any(k in text for k in NEG_KEYWORDS):
        glow = 0
    elif any(k in text for k in GLOW_KEYWORDS):
        glow = 1
    else:
        return None

    if window is None:
        # 朝霞关键词优先判断时段
        window = "morning" if ("朝霞" in text or "日出" in text) else "evening"
    return window, glow


def append_posts(path, rows):
    """向 posts.csv 追加观测记录（字段缺失自动补齐）。"""
    fieldnames = ["date", "window", "glow", "source", "note"]
    need_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def create_sample_posts(path):
    """首次运行时生成一个示例 posts.csv，便于用户理解格式。"""
    if os.path.exists(path):
        return
    sample = [
        {"date": "2026-09-12", "window": "evening", "glow": 1,
         "source": "weibo", "note": "西安晚霞刷屏，火烧云"},
        {"date": "2026-09-13", "window": "evening", "glow": 0,
         "source": "manual", "note": "阴天，未见晚霞"},
    ]
    append_posts(path, sample)
