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
import time

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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


def _request(url, params, timeout=30, retries=2):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
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
