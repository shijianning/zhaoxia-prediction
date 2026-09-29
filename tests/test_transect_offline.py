"""离线回归测试：方向性评分曲线 + 降级一致性 + 限流熔断。

不需要网络，任何时候都能跑：

    python tests/test_transect_offline.py

重点保护两类"容易在重构中被悄悄破坏"的性质：

1. **降级一致性**：没有剖面数据（未启用 / 取数失败）时，评分必须与升级前的
   7 因子版本**逐位一致**。这条一旦破了，历史评分和新评分就不可比了。
2. **因子不适用时是"整体移出加权"，不是"给中间分"**：带权重的中间分依然会
   牵动总分，那不叫不表态。
"""
import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import glow_rules as gr
from src import solar
from src import weather
from src.transect import TRANSECT_DISTANCES_KM as D

_FAILED = []


def check(label, got, want, tol=1e-6):
    good = (got == want) if isinstance(want, (str, type(None))) else (
        got is not None and abs(got - want) <= tol)
    if not good:
        _FAILED.append(label)
    print(f"  {'OK  ' if good else 'FAIL'} {label}: got={got} want={want}")


# 升级前的 7 因子权重（降级时必须复现的就是这组）
W7 = {"云结构": 0.30, "低云遮挡": 0.12, "湿度": 0.12, "降水": 0.10,
      "风速": 0.08, "通透度": 0.15, "能见度": 0.13}


def test_boundary_curve():
    print("\n[1] 剖面云边界曲线")
    assert D[13] == 380 and D[14] == 420 and D[-1] == 500
    check("全 0 剖面 -> km=None（不表态）",
          gr.boundary_distance_score([0.0] * len(D), D)[0], None)
    check("全 0 剖面 -> score=None（不表态）",
          gr.boundary_distance_score([0.0] * len(D), D)[1], None)
    km, sc = gr.boundary_distance_score([60.0] * 13 + [0.0] * 4, D)
    check("边缘 380km -> km", km, 380.0)
    check("边缘 380km -> score", sc, 95.0)
    km, sc = gr.boundary_distance_score([60.0] * 14 + [0.0] * 3, D)
    check("边缘 420km -> score", sc, 80.0)
    km, sc = gr.boundary_distance_score([60.0] * 3 + [0.0] * 14, D)
    check("边缘 60km -> km", km, 60.0)
    check("边缘 60km -> score", sc, 15.0)
    km, sc = gr.boundary_distance_score([80.0] * len(D), D)
    check("全程有云(无边缘) -> km", km, 500.0)
    check("全程有云(无边缘) -> score", sc, 0.0)


def test_block_curve():
    print("\n[2] 太阳方位遮挡曲线")
    check("近场低云 0% -> 100", gr.azimuth_block_score(0.0), 100.0)
    check("近场低云 50% -> 55", gr.azimuth_block_score(50.0), 55.0)
    check("近场低云 100% -> 10", gr.azimuth_block_score(100.0), 10.0)
    check("无数据 -> None", gr.azimuth_block_score(None), None)


CASES = [
    {"cloud_high": 40, "cloud_mid": 30, "cloud_low": 10, "humidity": 55,
     "wind": 4, "precip": 0, "aod": 0.2, "visibility": 18000},
    {"cloud_high": 0, "cloud_mid": 0, "cloud_low": 90, "humidity": 95,
     "wind": 14, "precip": 3, "aod": 1.1, "visibility": 800},
    {"cloud_high": 60, "cloud_mid": 20, "cloud_low": 5, "humidity": 50,
     "wind": 3, "precip": 0, "aod": None, "visibility": None},
]


def test_degradation_parity():
    """最重要的性质：没有剖面数据时必须与旧版逐位一致。"""
    print("\n[3] 降级一致性（无剖面数据 == 旧版 7 因子）")
    for i, f in enumerate(CASES, 1):
        score, bd = gr.rule_score(f)
        manual = sum(W7[k] * bd[k] for k in W7)
        check(f"case{i} 命中手算 7 因子", score, round(manual, 1))
        check(f"case{i} 空 transect 无影响",
              gr.rule_score(dict(f, transect={}))[0], score)
        check(f"case{i} transect 为 None 无影响",
              gr.rule_score(dict(f, transect=None))[0], score)


def test_weighting():
    print("\n[4] 有剖面数据时的加权与归一化")
    f = dict(CASES[0], transect={"boundary_score": 95.0, "block_score": 100.0})
    score, bd = gr.rule_score(f)
    base = sum(W7[k] * bd[k] for k in W7)
    check("9 因子加权", score, round((0.84 * base + 0.09 * 95.0 + 0.07 * 100.0), 1))
    check("breakdown 含剖面云边界", "剖面云边界" in bd, True)
    check("breakdown 含太阳方位遮挡", "太阳方位遮挡" in bd, True)

    # 只有遮挡因子可用 -> 剩余权重按 0.91 归一
    f = dict(CASES[0], transect={"boundary_score": None, "block_score": 100.0})
    score2, bd2 = gr.rule_score(f)
    base2 = sum(W7[k] * bd2[k] for k in W7)
    check("仅一项可用时按 0.91 归一", score2, round((0.84 * base2 + 7.0) / 0.91, 1))
    check("此时不含剖面云边界", "剖面云边界" in bd2, False)


def test_factor_note():
    print("\n[5] 最差因子提示")
    note = gr.factor_note({"云结构": 90, "低云遮挡": 88, "湿度": 85, "降水": 100,
                           "风速": 95, "通透度": 92, "能见度": 90,
                           "剖面云边界": 0.0, "太阳方位遮挡": 95})
    check("识别出剖面云边界", "云层边缘" in note, True)


def test_geometry():
    print("\n[6] 航点几何与阴影高度角")
    lat, lon = 34.34, 108.94
    l2, n2 = solar.destination_point(lat, lon, 90, 500)
    check("正东 500km 经度", round(n2, 1), 114.4, tol=0.05)
    check("正东 500km 纬度基本不变", round(l2, 1), 34.2, tol=0.15)
    check("5km 层阴影角", round(solar.shadow_height_deg(5, lat), 2), 2.27, tol=0.01)
    check("10km 层阴影角", round(solar.shadow_height_deg(10, lat), 2), 3.21, tol=0.01)
    # 日出日落时刻的太阳高度角应恰好是 -0.833°（二分精修的意义）
    pos = solar.event_position(lat, lon, "2026-09-29T18:31", 28800)
    check("日落时刻太阳高度角", round(pos["elevation"], 2), -0.83, tol=0.02)


def test_rate_limit_breaker():
    """429 必须立刻熔断，且冷却期内不再发出任何网络请求。

    这是防止"被限流后重试放大请求量、把分钟级限流升级成整点封禁"的关键保护。
    """
    print("\n[7] 限流熔断")
    calls = {"n": 0}

    class FakeResp:
        status_code = 429

        def raise_for_status(self):
            import requests
            exc = requests.HTTPError("429 Too Many Requests")
            exc.response = self
            raise exc

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return FakeResp()

    cfg = {"city": {"latitude": 1, "longitude": 2, "timezone": "UTC"},
           "forecast": {"days": 1}}

    cooldown_before = weather.cooldown_remaining()
    with mock.patch("src.weather.requests.get", fake_get):
        try:
            weather.get_forecast(cfg)
            check("首次 429 应抛 RateLimited", "未抛异常", "RateLimited")
        except weather.RateLimited:
            check("首次 429 抛出 RateLimited", True, True)
        check("429 不重试（只发 1 次请求）", calls["n"], 1)
        check("已进入冷却期", weather.in_cooldown(), True)

        before = calls["n"]
        try:
            weather.get_forecast(cfg)
            check("冷却期内应快速失败", "未拦截", "RateLimited")
        except weather.RateLimited:
            check("冷却期内快速失败", True, True)
        check("冷却期内零网络请求", calls["n"], before)

    # 复原全局冷却状态，避免影响同进程内的其他测试
    import src.weather as w
    with w._cooldown_lock:
        w._cooldown_until[0] = time.monotonic() + max(cooldown_before, 0.0)


def main():
    for fn in (test_boundary_curve, test_block_curve, test_degradation_parity,
               test_weighting, test_factor_note, test_geometry,
               test_rate_limit_breaker):
        fn()
    print("\n" + ("=" * 46))
    if _FAILED:
        print(f"失败 {len(_FAILED)} 项：")
        for item in _FAILED:
            print("  -", item)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
