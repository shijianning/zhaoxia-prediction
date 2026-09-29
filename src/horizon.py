"""地形遮蔽 —— PVGIS 地平线剖面（免费、无需 API Key）。

平原上地平线就是水平的（仰角 0°）；但山区城市西侧若有群山，
地平线会被抬到 1~5°，意味着**观察者能看到的低空天空被切掉一块** ——
而低角度火烧云恰恰就出现在那片被切掉的天空里。

本模块从欧盟联合研究中心（JRC）的 PVGIS 服务拉取 90m 分辨率 DEM 算出的
地平线剖面（49 个方位 × 7.5° 间隔），按太阳方位插值出该方向的
地平线仰角 `terrain_deg`，交给 `glow_rules.terrain_score()` 打分。

接口：GET https://re.jrc.ec.europa.eu/api/v5_3/printhorizon?lat=&lon=&outputformat=json
返回 outputs.horizon_profile: [{"A": 方位角(度, -180~180), "H_hor": 仰角(度)}, ...]

设计要点：
- **磁盘缓存**：地平线是静态地形量，一座城市只算一次，之后永久复用。
  既减少请求，也符合 PVGIS 的使用条款。
- **优雅降级**：PVGIS 未覆盖该点 / 网络失败时返回 None，
  由 `glow_rules` 把「地形遮蔽」因子整体移出加权，不影响其余评分。
"""
import json
import os
import time

import requests

PVGIS_HORIZON_URL = "https://re.jrc.ec.europa.eu/api/v5_3/printhorizon"

# 进程内缓存：全国地图逐城调用时避免重复读盘
_MEM_CACHE = {}


def enabled(cfg):
    """是否启用地形遮蔽（默认开）。"""
    return bool((cfg.get("horizon") or {}).get("enabled", True))


def _cache_dir(cfg):
    d = (cfg.get("horizon") or {}).get("cache_dir") or "data/horizon"
    return d


def _cache_path(cfg, lat, lon):
    return os.path.join(_cache_dir(cfg), f"{float(lat):.4f}_{float(lon):.4f}.json")


def _normalize(raw_profile):
    """把 PVGIS 的 [{A, H_hor}] 转成 [(azimuth_0_360, altitude), ...]。"""
    out = []
    for item in raw_profile or []:
        try:
            az = float(item["A"]) % 360.0
            alt = float(item["H_hor"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((az, alt))
    return sorted(out)


def fetch_horizon(cfg, lat, lon, timeout=40, retries=2):
    """拉取（或从缓存读取）地平线剖面。

    返回 [(azimuth_0_360, altitude_deg), ...]；不可用时返回 None。
    网络偶发失败会重试 `retries` 次（PVGIS 首次 TLS 握手偶尔偏慢/被拒）。
    """
    path = _cache_path(cfg, lat, lon)
    mem_key = f"{float(lat):.4f}_{float(lon):.4f}"
    if mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            prof = _normalize(cached.get("profile"))
            if prof:
                _MEM_CACHE[mem_key] = prof
                return prof
        except Exception:
            pass  # 缓存损坏 → 重新拉取

    prof = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                PVGIS_HORIZON_URL,
                params={"lat": lat, "lon": lon, "outputformat": "json"},
                timeout=timeout,
                headers={"User-Agent": "zhaoxia-prediction/1.0 (sunset glow forecast)"},
            )
            resp.raise_for_status()
            payload = resp.json()
            raw = (payload.get("outputs") or {}).get("horizon_profile") or []
            prof = _normalize(raw)
            if prof:
                break
        except Exception:
            prof = None
        if attempt < retries:
            time.sleep(1.0)

    if not prof:
        return None

    _MEM_CACHE[mem_key] = prof
    try:
        os.makedirs(_cache_dir(cfg), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"lat": float(lat), "lon": float(lon),
                       "profile": [{"A": a, "H_hor": h} for a, h in prof]},
                      f, ensure_ascii=False)
    except Exception:
        pass  # 缓存写失败不影响本次结果

    return prof


def interpolate(profile, azimuth_deg):
    """在环形地平线剖面上按方位角线性插值，返回地平线仰角（度）。

    profile: [(azimuth_0_360, altitude), ...]；空则返回 None。
    """
    if not profile:
        return None
    pts = sorted((float(a) % 360.0, float(h)) for a, h in profile)
    if len(pts) == 1:
        return pts[0][1]

    base = pts[0][0]
    # 把目标方位搬到 [base, base+360) 区间，配合下面的环形扩展
    az = base + ((float(azimuth_deg) % 360.0 - base) % 360.0)
    ext = pts + [(base + 360.0, pts[0][1])]

    for (a1, h1), (a2, h2) in zip(ext, ext[1:]):
        if a1 <= az <= a2:
            if a2 <= a1:
                return h1
            t = (az - a1) / (a2 - a1)
            return h1 + t * (h2 - h1)
    return pts[-1][1]


def horizon_altitude(cfg, lat, lon, azimuth_deg, timeout=40):
    """一步得到：太阳方位上的地形地平线仰角（度）。不可用时返回 None。"""
    if not enabled(cfg) or azimuth_deg is None:
        return None
    prof = fetch_horizon(cfg, lat, lon, timeout=timeout)
    if not prof:
        return None
    return interpolate(prof, azimuth_deg)
