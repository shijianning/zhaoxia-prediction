"""配置加载模块。

读取项目根目录的 config.yaml，并与内置默认值合并，保证字段齐全。
"""
import copy
import os

import yaml

# 项目根目录 = 本文件所在目录的上一级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "city": {
        "name": "西安",
        "latitude": 34.34,
        "longitude": 108.94,
        "timezone": "Asia/Shanghai",
    },
    "forecast": {"days": 7},
    "model": {
        "type": "logistic",
        "bootstrap": True,
        "min_samples": 30,
        "model_path": "data/model/glow_model.joblib",
    },
    "data": {
        "posts_csv": "data/raw/posts.csv",
        "history_days": 120,
    },
    "third_party": {
        "geovisearth": {
            "enabled": False,
            "token": "",
            "base_url": "https://api.open.geovisearth.com/v2/grid/glow/day",
            "mete_codes": "glow,aod",
            "level": True,
        },
    },
    "database": {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "zhaoxia",
    },
    "notify": {
        "enabled": False,
        "sckey": "",
        "threshold": 70,
        "morning_threshold": None,
        "evening_threshold": None,
    },
    "output": {
        "dir": "output",
        "report": "output/report.html",
    },
    "map": {
        "tmap_key": "",
    },
}


def _merge(dst, src):
    """递归合并，src 覆盖 dst。"""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def load_config(path=None):
    """加载配置，返回完整配置字典。

    优先级：config.yaml（默认）< config.local.yaml（本地覆盖，不入库）。
    敏感信息（如 notify.sckey、database.password、星图云 token）建议写在
    config.local.yaml 中，该文件已被 .gitignore 排除，不会提交到仓库。
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        _merge(cfg, user_cfg)

    # 本地覆盖（可放密钥，不入库）
    local_path = os.path.join(BASE_DIR, "config.local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        _merge(cfg, local_cfg)

    # 相对路径统一解析为基于项目根目录的绝对路径
    cfg["model"]["model_path"] = os.path.join(BASE_DIR, cfg["model"]["model_path"])
    cfg["data"]["posts_csv"] = os.path.join(BASE_DIR, cfg["data"]["posts_csv"])
    cfg["output"]["dir"] = os.path.join(BASE_DIR, cfg["output"]["dir"])
    cfg["output"]["report"] = os.path.join(BASE_DIR, cfg["output"]["report"])
    return cfg
