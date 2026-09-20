"""统一日志模块。

把脚本运行情况写到 output/logs/ 下按日滚动的日志文件，同时输出到控制台。
定时任务（schtasks）跑起来后没有可见终端，日志文件是排查问题的唯一依据。
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

_LOGGER_NAME = "zhaoxia"


def setup_logging(cfg):
    """初始化日志，返回 logger。重复调用幂等（不会叠加 handler）。

    cfg: load_config() 的返回值。
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = os.path.join(cfg["output"]["dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # 文件：按天轮转，保留 30 份
    fh = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


def get_logger():
    """获取已初始化的 logger（未初始化时返回一个最小 logger）。"""
    return logging.getLogger(_LOGGER_NAME)
