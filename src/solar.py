"""太阳位置与方位几何计算（纯标准库，无第三方依赖）。

朝霞/晚霞是有方向性的光学现象，但很容易被当成"看头顶云量"来算。
本模块提供三个做方向性评分必须的几何量：

1. `solar_position()` —— 某一时刻太阳的方位角与高度角。
   光来自太阳所在方位，所以"朝向太阳的那半个天空"比"全天空平均"更有判断价值。
2. `destination_point()` —— 从某点沿指定方位角前进 d 公里后的坐标。
   用于沿太阳方位拉一条剖面（见 transect.py）。
3. `shadow_height_deg()` —— 某高度的云层"还在地球曲率阴影之外、能被低角度阳光
   照亮"所允许的最大太阳下沉角。日落瞬间太阳约在地平线下 0.83°，据此可判断
   哪个高度的云还能接光、哪个高度的云已经变黑。

算法采用 NOAA Solar Calculator 的公开公式，精度约 ±0.5°（时间上约 ±1 分钟），
对"哪个方向出霞""哪个高度还能被照亮"这种量级的判断完全够用，且不引入 ephem /
pysolar 之类的额外依赖。
"""
import math

# WGS-84 椭球半轴（用于按纬度修正地球半径）
_R_EQ_KM = 6378.1370
_R_POL_KM = 6356.7523

# 标准日出日落时，太阳中心在地平线下的角度
# （太阳视半径 0.267° + 地面大气折射约 0.566°）
SUNRISE_ELEVATION = -0.833


def earth_radius_km(latitude):
    """返回该纬度处的地球曲率半径（公里）。

    高纬度处地球更扁，用固定 6371km 会让几百公里外的剖面点产生
    几十公里的位置误差，进而错采到别的天气格点，因此按纬度修正。
    """
    phi = math.radians(latitude)
    a2 = (_R_EQ_KM * math.cos(phi)) ** 2
    b2 = (_R_POL_KM * math.sin(phi)) ** 2
    return math.sqrt((_R_EQ_KM ** 2 * a2 + _R_POL_KM ** 2 * b2) / (a2 + b2))


def _julian_day(when_utc):
    """UTC datetime → 儒略日。"""
    year, month = when_utc.year, when_utc.month
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    day_fraction = (
        when_utc.day
        + (when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0) / 24.0
    )
    return (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day_fraction
        + b
        - 1524.5
    )


def solar_position(latitude, longitude, when_utc):
    """计算给定时刻太阳的方位角与高度角（单位：度）。

    latitude/longitude: 度；when_utc: 必须是 UTC 时间（naive，表示 UTC）。
    返回 {"azimuth": 0-360（正北起顺时针）, "elevation": 高度角（负值=在地平线下）}。

    方位角是"太阳在哪个方向"，高度角是"太阳有多高"。
    """
    jd = _julian_day(when_utc)
    t = (jd - 2451545.0) / 36525.0

    # 太阳几何平黄经与平近点角
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anomaly = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    m_rad = math.radians(mean_anomaly)
    # 中心差 → 真黄经
    equation_of_center = (
        math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m_rad) * 0.000289
    )
    true_longitude = l0 + equation_of_center

    # 视黄经与黄赤交角
    omega = 125.04 - 1934.136 * t
    apparent_longitude = (
        true_longitude - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    )
    mean_obliquity = (
        23.0
        + 26.0 / 60.0
        + 21.448 / 3600.0
        - t * (46.8150 / 3600.0)
        + t ** 2 * (0.00059 / 3600.0)
        - t ** 3 * (0.001813 / 3600.0)
    )
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(omega))

    # 赤纬
    declination = math.degrees(
        math.asin(
            math.sin(math.radians(obliquity))
            * math.sin(math.radians(apparent_longitude))
        )
    )

    # 时差（分钟）→ 真太阳时 → 时角
    y = math.tan(math.radians(obliquity / 2.0)) ** 2
    equation_of_time = 4.0 * math.degrees(
        y * math.sin(2 * math.radians(l0))
        - 2 * eccentricity * math.sin(m_rad)
        + 4 * eccentricity * y * math.sin(m_rad) * math.cos(2 * math.radians(l0))
        - 0.5 * y * y * math.sin(4 * math.radians(l0))
        - 1.25 * eccentricity * eccentricity * math.sin(2 * mean_anomaly)
    )
    utc_minutes = when_utc.hour * 60 + when_utc.minute + when_utc.second / 60.0
    true_solar_time = (utc_minutes + equation_of_time + 4.0 * longitude) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0

    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)

    cos_zenith = (
        math.sin(lat_rad) * math.sin(dec_rad)
        + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    geometric_elevation = 90.0 - math.degrees(math.acos(cos_zenith))

    # 大气折射修正（NOAA 分段公式）
    if geometric_elevation > 85.0:
        refraction = 0.0
    elif geometric_elevation > 5.0:
        tan_e = math.tan(math.radians(geometric_elevation))
        refraction = 58.1 / tan_e - 0.07 / tan_e ** 3 + 0.000086 / tan_e ** 5
    elif geometric_elevation > -0.575:
        e = geometric_elevation
        refraction = 1735 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    else:
        refraction = -20.772 / math.tan(math.radians(geometric_elevation))
    elevation = geometric_elevation + refraction / 3600.0

    azimuth = math.degrees(
        math.atan2(
            math.sin(ha_rad),
            math.cos(ha_rad) * math.sin(lat_rad) - math.tan(dec_rad) * math.cos(lat_rad),
        )
    )
    azimuth = (azimuth + 180.0) % 360.0

    return {"azimuth": azimuth, "elevation": elevation}


def destination_point(latitude, longitude, bearing_deg, distance_km, radius_km=None):
    """从 (latitude, longitude) 沿 bearing_deg 方位角前进 distance_km 后的坐标。

    球面大圆前进公式（与 Google/OSM 的 destination point 一致）。
    返回 (lat, lon)，单位度。
    """
    r = radius_km if radius_km else earth_radius_km(latitude)
    angular = distance_km / r
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    # 归一化到 -180~180，避免跨越日期变更线时坐标越界
    lon2 = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return math.degrees(lat2), lon2


def shadow_height_deg(height_km, latitude):
    """某高度层"还能被低角度阳光照亮"所允许的最大太阳下沉角（度）。

    地球是圆的：太阳落到地平线以下后，越低的云越早进入地球本身的阴影。
    位于高度 h 的云层被照亮的前提是太阳下沉角 θ 满足
        θ <= arccos(R / (R + h))
    这就是"阴影高度角"，也叫地平线下沉角。

    例（北纬 34°）：h=1km → 约 1.02°；h=5km → 约 2.27°；h=10km → 约 3.22°。
    日落瞬间太阳约在下沉 0.83°，因此 1km 的低云此时已经变暗，
    而 5km 以上的中高云仍在接光——这正是"火烧云多是中高云"的几何原因。
    """
    if height_km is None or height_km <= 0:
        return 0.0
    r = earth_radius_km(latitude)
    ratio = r / (r + float(height_km))
    if ratio >= 1.0:
        return 0.0
    return math.degrees(math.acos(ratio))


def elevation_at_utc_offset(local_iso, utc_offset_seconds):
    """把 Open-Meteo 返回的本地时刻字符串转成 UTC datetime。

    Open-Meteo 的 daily.sunrise/sunset 是本地时区的 naive 字符串
    （如 "2026-09-29T06:36"），同时响应里带 `utc_offset_seconds`。
    用它换算可以避免引入 tzdata / zoneinfo 依赖。
    """
    from datetime import datetime, timedelta

    naive = datetime.fromisoformat(local_iso)
    return naive - timedelta(seconds=int(utc_offset_seconds or 0))


def event_position(latitude, longitude, local_iso, utc_offset_seconds):
    """给出日出/日落那一刻的太阳位置（方位角 + 高度角）。

    Open-Meteo 的日出日落时间只精确到分钟，直接用它算方位角会有约 1° 误差
    （日出日落附近方位角变化约 0.2~0.3°/分钟）。这里以它为初值，用二分法
    精修到"太阳中心恰好位于地平线下 0.833°"的真实时刻，再求方位角。

    返回 {"azimuth", "elevation", "utc"}；精修失败时退回初值，不抛异常。
    """
    from datetime import timedelta

    approx = elevation_at_utc_offset(local_iso, utc_offset_seconds)

    lo = approx - timedelta(minutes=10)
    hi = approx + timedelta(minutes=10)
    f_lo = solar_position(latitude, longitude, lo)["elevation"] - SUNRISE_ELEVATION
    f_hi = solar_position(latitude, longitude, hi)["elevation"] - SUNRISE_ELEVATION

    best = approx
    if f_lo == 0.0:
        best = lo
    elif f_hi == 0.0:
        best = hi
    elif f_lo * f_hi < 0.0:
        # 二分 20 次 → 时间精度优于 0.01 秒，远超需要
        a, b, fa = lo, hi, f_lo
        for _ in range(20):
            mid = a + (b - a) / 2
            fm = solar_position(latitude, longitude, mid)["elevation"] - SUNRISE_ELEVATION
            if fa * fm <= 0.0:
                b = mid
            else:
                a, fa = mid, fm
        best = a + (b - a) / 2

    pos = solar_position(latitude, longitude, best)
    pos["utc"] = best
    return pos
