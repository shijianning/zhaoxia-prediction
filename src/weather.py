"""天气数据获取模块。

使用 Open-Meteo 免费接口（无需 API Key）：
- 预报接口: api.open-meteo.com/v1/forecast
- 历史接口: archive-api.open-meteo.com/v1/archive

返回的逐小时变量：
cloud_cover / cloud_cover_low / cloud_cover_mid / cloud_cover_high
relative_humidity_2m / wind_speed_10m / precipitation / temperature_2m / weather_code
（预报额外包含 precipitation_probability）
以及每日 sunrise / sunset。
"""
import threading
import time

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---- 全局限流 -------------------------------------------------------------
# Open-Meteo 免费额度按"每分钟调用数"限流，超限直接返回 429。启用太阳方位剖面
# 后每城请求数翻倍（主预报 + 剖面），全国地图再叠加 8 路并发，实测会撞上限流。
# 这里在进程内做轻量节流，保证任意两次请求间隔不低于 _MIN_INTERVAL 秒；
# 配合 _request 里对 429 的加长退避，多城批量抓取会明显更稳。
_MIN_INTERVAL = 0.15
_rate_lock = threading.Lock()
_last_request_at = [0.0]


def _throttle():
    with _rate_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL - (now - _last_request_at[0])
        if wait > 0:
            time.sleep(wait)
        _last_request_at[0] = time.monotonic()


# ---- 限流熔断 -------------------------------------------------------------
# Open-Meteo 免费额度除"每分钟"上限外还有**每小时**上限，超限后返回 429：
#   {"reason":"Hourly API request limit exceeded. Please try again in the next hour."}
#
# 真正危险的不是被限流，而是**被限流之后还在重试**：重试会把请求量成倍放大
# （单城 2 次 × 每请求 4 次尝试 × 整批 3 轮重试 = 24 倍），几分钟内就能把一次
# 轻微的分钟级限流升级成整点封禁，让当天的定时任务全部失效。
#
# 因此这里做熔断：一旦收到 429，立刻进入冷却期；冷却期内所有请求**直接快速失败、
# 不发网络请求**，让上层批量逻辑尽快收敛退出。批量入口在重试前会查
# `in_cooldown()`，冷却中就不再重试。
DEFAULT_COOLDOWN_SECONDS = 900.0   # 15 分钟
_cooldown_until = [0.0]
_cooldown_lock = threading.Lock()


class RateLimited(RuntimeError):
    """触发 Open-Meteo 限流。上层应当停止批量抓取并退出，而不是重试。"""


def in_cooldown():
    """是否处于限流冷却期（冷却期内调用方不应再发起批量请求）。"""
    with _cooldown_lock:
        return time.monotonic() < _cooldown_until[0]


def cooldown_remaining():
    """冷却期剩余秒数，0 表示未在冷却。"""
    with _cooldown_lock:
        return max(0.0, _cooldown_until[0] - time.monotonic())


def _start_cooldown(seconds=DEFAULT_COOLDOWN_SECONDS):
    with _cooldown_lock:
        _cooldown_until[0] = max(_cooldown_until[0], time.monotonic() + seconds)

# 历史与预报共用的逐小时变量
HOURLY_VARS = [
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "temperature_2m",
    "weather_code",
    "visibility",   # 水平能见度(米)，用于"大气通透"因子（融合 sunset-prediction 5因子模型）
]

# 每日天气概览变量（报告展示：日出日落 / 天气现象 / 最高最低温 / 全天降水）
DAILY_WEATHER_VARS = [
    "sunrise",
    "sunset",
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
]


def _request(url, params, timeout=30, retries=3):
    if in_cooldown():
        raise RateLimited(
            f"Open-Meteo 限流冷却中（剩余 {cooldown_remaining():.0f}s），跳过请求"
        )

    last_exc = None
    for attempt in range(retries + 1):
        _throttle()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                # 限流。退避重试只会放大请求量，所以直接熔断并抛出，
                # 由批量入口（national_map.run_national）识别后停止重试。
                _start_cooldown()
                raise RateLimited(f"Open-Meteo 限流（429）：{exc}") from exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc


def get_forecast(cfg, days=None):
    """获取未来预报数据。"""
    city = cfg["city"]
    days = days or cfg["forecast"]["days"]
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": ",".join(HOURLY_VARS + ["precipitation_probability"]),
        "daily": ",".join(DAILY_WEATHER_VARS),
        "timezone": city["timezone"],
        "forecast_days": days,
    }
    return _request(FORECAST_URL, params)


def get_historical(cfg, start_date, end_date):
    """获取历史数据（用于训练）。start_date/end_date 为 'YYYY-MM-DD' 字符串。"""
    city = cfg["city"]
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "daily": "sunrise,sunset",
        "timezone": city["timezone"],
    }
    return _request(ARCHIVE_URL, params)


# 太阳方位剖面用到的变量（只需要分层云量，响应体小、速度快）
TRANSECT_HOURLY_VARS = [
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
]


def get_transect(cfg, lats, lons, days=None):
    """沿太阳方位角的一次多坐标剖面请求。

    Open-Meteo 支持在单个请求里给多组经纬度（逗号分隔），返回一个列表。
    本项目借这个能力把早、晚两条射线（各 17 点）合并成一次调用，
    因此"方向性评分"不增加任何请求次数以外的成本。

    lats/lons: 等长的坐标序列。
    返回：统一为 list 的各点响应（单点时 Open-Meteo 返回 dict，这里也包成 list）。
    """
    city = cfg["city"]
    days = days or cfg["forecast"]["days"]
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lats),
        "longitude": ",".join(f"{v:.4f}" for v in lons),
        "hourly": ",".join(TRANSECT_HOURLY_VARS),
        "timezone": city["timezone"],
        "forecast_days": days,
    }
    payload = _request(FORECAST_URL, params, timeout=60)
    return payload if isinstance(payload, list) else [payload]


# 用于交叉验证的多个独立全球气象模式
CROSSCHECK_MODELS = ["gfs_global", "icon_global", "gem_global"]


def get_multi_model_forecast(cfg, days=None, models=None):
    """拉取多个独立气象模型的预报，用于交叉验证。

    返回的 hourly 中，每个变量会按模型拆分为 `<变量>_<模型>` 字段。
    """
    city = cfg["city"]
    days = days or cfg["forecast"]["days"]
    models = models or CROSSCHECK_MODELS
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": ",".join(HOURLY_VARS),
        "daily": "sunrise,sunset",
        "timezone": city["timezone"],
        "forecast_days": days,
        "models": ",".join(models),
    }
    return _request(FORECAST_URL, params, timeout=60)


# 空气质量接口（气溶胶光学厚度 AOD / PM2.5）
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
AIR_QUALITY_VARS = ["aerosol_optical_depth", "pm2_5"]


def get_air_quality(cfg, days=None):
    """获取未来空气质量(气溶胶 AOD / PM2.5)预报。"""
    city = cfg["city"]
    days = days or cfg["forecast"]["days"]
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": ",".join(AIR_QUALITY_VARS),
        "daily": "sunrise,sunset",
        "timezone": city["timezone"],
        "forecast_days": days,
    }
    return _request(AIR_QUALITY_URL, params, timeout=60)


def get_historical_air_quality(cfg, start_date, end_date):
    """获取历史空气质量数据（用于训练）。"""
    city = cfg["city"]
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_VARS),
        "daily": "sunrise,sunset",
        "timezone": city["timezone"],
    }
    return _request(AIR_QUALITY_URL, params, timeout=60)
