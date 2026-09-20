"""特征工程模块。

把逐小时天气数据，在每天日出/日落窗口处抽取为特征字典。

窗口定义：
- 朝霞(morning): 日出前后，取 [日出时刻-1h, 日出时刻+1h] 的均值/极值
- 晚霞(evening): 日落前后，取 [日落时刻-1h, 日落时刻+1h] 的均值/极值

输出特征键：
cloud_cover / cloud_low / cloud_mid / cloud_high / humidity / wind /
temp / precip / precip_prob / weather_code / day_of_year
"""
import datetime as dt


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _smax(vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else 0.0


def _window_features(hourly, idx, day_str):
    n = len(hourly["time"])
    lo = max(0, idx - 1)
    hi = min(n - 1, idx + 1)

    def mean(var):
        return _mean(hourly.get(var, [])[lo:hi + 1])

    def smax(var):
        return _smax(hourly.get(var, [])[lo:hi + 1])

    def mean_or_none(var):
        vals = hourly.get(var, [])[lo:hi + 1]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    day_of_year = dt.datetime.strptime(day_str, "%Y-%m-%d").timetuple().tm_yday

    return {
        "cloud_cover": mean("cloud_cover"),
        "cloud_low": mean("cloud_cover_low"),
        "cloud_mid": mean("cloud_cover_mid"),
        "cloud_high": mean("cloud_cover_high"),
        "humidity": mean("relative_humidity_2m"),
        "wind": mean("wind_speed_10m"),
        "temp": mean("temperature_2m"),
        "precip": smax("precipitation"),
        "precip_prob": smax("precipitation_probability"),
        "weather_code": int(smax("weather_code")),
        "visibility": mean_or_none("visibility"),  # 缺失时为 None（中性分），避免误判
        "day_of_year": day_of_year,
    }


def extract_day_features(data):
    """把 Open-Meteo 返回的数据转换为 {(date, window): feature_dict}。

    data: get_forecast / get_historical 的返回值。
    window ∈ {"morning", "evening"}。
    """
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    times = hourly.get("time", [])
    # 建立 "YYYY-MM-DDTHH" -> 索引 的映射
    key_to_idx = {t[:13]: i for i, t in enumerate(times)}

    features = {}
    for i, day in enumerate(daily.get("time", [])):
        for window, tstr in (("morning", daily["sunrise"][i]),
                             ("evening", daily["sunset"][i])):
            key = tstr[:13]
            idx = key_to_idx.get(key)
            if idx is None:
                continue
            features[(day, window)] = _window_features(hourly, idx, day)
    return features


def add_air_quality(day_feats, aq_data):
    """把空气质量数据（气溶胶 AOD / PM2.5）合并进每个窗口特征。

    day_feats: extract_day_features 的返回值 {(date, window): feature}。
    aq_data: weather.get_air_quality 的返回值。
    合并后每个 feature 增加 aod / pm2_5 字段（取不到时为 None）。
    """
    hourly = aq_data.get("hourly", {})
    daily = aq_data.get("daily", {})
    times = hourly.get("time", [])
    key_to_idx = {t[:13]: i for i, t in enumerate(times)}

    def mean(var, lo, hi):
        vals = hourly.get(var, [])[lo:hi + 1]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    n_days = len(daily.get("time", []))
    for i, day in enumerate(daily.get("time", [])):
        sr = daily.get("sunrise", [None] * n_days)[i]
        ss = daily.get("sunset", [None] * n_days)[i]
        for window, tstr in (("morning", sr), ("evening", ss)):
            if not tstr:
                continue
            f = day_feats.get((day, window))
            if f is None:
                continue
            idx = key_to_idx.get(tstr[:13])
            if idx is None:
                continue
            lo = max(0, idx - 1)
            hi = min(len(times) - 1, idx + 1)
            f["aod"] = mean("aerosol_optical_depth", lo, hi)
            f["pm2_5"] = mean("pm2_5", lo, hi)


def extract_daily_weather(data):
    """提取每日天气概览，供报告展示。

    data: weather.get_forecast 的返回值（其 daily 需包含 DAILY_WEATHER_VARS）。
    返回 {date: {code, tmax, tmin, precip_sum, precip_prob, sunrise, sunset}}，
    字段取不到时为 None。
    """
    daily = data.get("daily", {})
    days = daily.get("time", [])
    n = len(days)

    def g(var):
        vals = daily.get(var, [])
        if len(vals) < n:
            vals = list(vals) + [None] * (n - len(vals))
        return vals

    code = g("weather_code")
    tmax = g("temperature_2m_max")
    tmin = g("temperature_2m_min")
    psum = g("precipitation_sum")
    pprob = g("precipitation_probability_max")
    sunrise = g("sunrise")
    sunset = g("sunset")

    result = {}
    for i, day in enumerate(days):
        result[day] = {
            "code": code[i],
            "tmax": tmax[i],
            "tmin": tmin[i],
            "precip_sum": psum[i],
            "precip_prob": pprob[i],
            "sunrise": sunrise[i],
            "sunset": sunset[i],
        }
    return result
