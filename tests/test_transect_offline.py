"""离线回归测试：评分曲线 + 缺失值语义 + 降级一致性 + 限流熔断 + 训练闸门。

不需要网络，任何时候都能跑：

    python tests/test_transect_offline.py

重点保护几类"容易在重构中被悄悄破坏"的性质：

1. **降级一致性**：因子齐全且无剖面数据时，评分必须与引入方向性之前的版本
   逐位一致；一旦破了，历史评分和新评分就不可比了。
2. **缺失数据的唯一语义**：值为 None 的因子必须**整体移出加权**并重归一化，
   而不是给一个"中间分" —— 带权重的中间分依然会牵动总分，那不叫不表态。
3. **真实标签闸门**：没有真实观测标注时绝不能启用 ML，因为弱监督标签是
   规则分的确定性函数，模型指标会虚高到 AUC≈1.0 却毫无预测意义。
"""
import os
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib

from src import crosscheck
from src import features as feat_mod
from src import glow_rules as gr
from src import horizon
from src import solar
from src import weather
from src.transect import TRANSECT_DISTANCES_KM as D

_FAILED = []


class _FakePipe:
    """只有 n_features_in_ 的假估计器，用于测试模型维度校验。

    必须定义在模块级，否则 pickle 找不到它（局部类无法序列化）。
    """

    def __init__(self, n):
        self.n_features_in_ = n


def check(label, got, want, tol=1e-6):
    good = (got == want) if isinstance(want, (str, type(None))) else (
        got is not None and abs(got - want) <= tol)
    if not good:
        _FAILED.append(label)
    print(f"  {'OK  ' if good else 'FAIL'} {label}: got={got} want={want}")


# 引入方向性之前的 7 因子权重（因子齐全时降级必须复现的就是这组）
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


def test_terrain_curve():
    print("\n[2b] 地形遮蔽曲线")
    check("无数据 -> None", gr.terrain_score(None), None)
    check("地平线 0° -> 100", gr.terrain_score(0.0), 100.0)
    check("地平线 2° -> 82", gr.terrain_score(2.0), 82.0)
    check("地平线 5° -> 55", gr.terrain_score(5.0), 55.0)
    check("地平线 20° 下限截断", gr.terrain_score(20.0), 0.0)


def test_aod_typing():
    print("\n[2c] AOD 分型（清洁支单调 / 沙尘支非单调）")
    check("无数据 -> None（移出加权）", gr.transparency_score(None), None)
    # 清洁支：单调递减
    check("清洁 0.1 -> 100", gr.transparency_score(0.1), 100.0)
    check("清洁 0.2 -> 88", gr.transparency_score(0.2), 88.0)
    check("清洁 0.5 -> 55", gr.transparency_score(0.5), 55.0)
    check("清洁 1.0 -> 18", gr.transparency_score(1.0), 18.0)
    # 沙尘支：峰值在 0.25，两端更低
    check("沙尘 0.1 -> 55", gr.transparency_score(0.1, 0.7), 55.0)
    check("沙尘 0.25 -> 100（峰值）", gr.transparency_score(0.25, 0.7), 100.0)
    check("沙尘 0.2 -> 85", gr.transparency_score(0.2, 0.7), 85.0)
    s_low = gr.transparency_score(0.1, 0.7)
    s_peak = gr.transparency_score(0.25, 0.7)
    s_high = gr.transparency_score(1.5, 0.7)
    check("沙尘支非单调（两端低于峰值）", s_low < s_peak and s_high < s_peak, True)
    # dust/pm10 比值
    check("dust_ratio 计算", gr.dust_ratio_of({"dust": 30, "pm10": 50}), 0.6)
    check("缺字段 -> None", gr.dust_ratio_of({"dust": 30}), None)
    check("pm10 为 0 -> None", gr.dust_ratio_of({"dust": 30, "pm10": 0}), None)


def test_precip_timing():
    print("\n[2d] 降水时序（雨洗效应）")
    check("无数据 -> 1.0", gr.washout_factor(None), 1.0)
    check("无雨 -> 1.0", gr.washout_factor(0.0), 1.0)
    check("微量降水 1mm -> 1.08（吸湿增长）", gr.washout_factor(1.0), 1.08)
    check("中雨 5mm -> 0.92", gr.washout_factor(5.0), 0.92)
    check("大雨 15mm -> 0.75（冲刷）", gr.washout_factor(15.0), 0.75)
    check("窗口有雨 -> 0", gr.precip_score(0.5, None), 0.0)
    check("窗口微量 -> 30", gr.precip_score(0.1, None), 30.0)
    check("窗口无雨/无前情 -> 100", gr.precip_score(0.0, None), 100.0)
    check("窗口无雨+前 3h 有雨 -> 88", gr.precip_score(0.0, 3.0), 88.0)
    check("窗口无雨+前日大雨 -> 100", gr.precip_score(0.0, 15.0), 100.0)


CASES = [
    {"cloud_high": 40, "cloud_mid": 30, "cloud_low": 10, "humidity": 55,
     "wind": 4, "precip": 0, "aod": 0.2, "visibility": 18000},
    {"cloud_high": 0, "cloud_mid": 0, "cloud_low": 90, "humidity": 95,
     "wind": 14, "precip": 3, "aod": 1.1, "visibility": 800},
    {"cloud_high": 60, "cloud_mid": 20, "cloud_low": 5, "humidity": 50,
     "wind": 3, "precip": 0, "aod": None, "visibility": None},
]


def test_degradation_parity():
    """最重要的性质：因子齐全且无剖面数据时，与旧版逐位一致。"""
    print("\n[3] 降级一致性（无剖面数据 == 旧版 7 因子）")
    for i, f in enumerate(CASES[:2], 1):
        score, bd = gr.rule_score(f)
        manual = sum(W7[k] * bd[k] for k in W7)
        check(f"case{i} 命中手算 7 因子", score, round(manual, 1))
        check(f"case{i} 空 transect 无影响",
              gr.rule_score(dict(f, transect={}))[0], score)
        check(f"case{i} transect 为 None 无影响",
              gr.rule_score(dict(f, transect=None))[0], score)
        check(f"case{i} 空 breakdown 的 transect 无影响",
              gr.rule_score(dict(f, transect={"boundary_score": None,
                                              "block_score": None,
                                              "terrain_score": None}))[0], score)


def test_missing_factor_removed():
    """缺数据的因子必须整整体移出加权并重归一化，而不是给中间分。"""
    print("\n[3b] 缺因子 = 移出加权（不是中性分）")
    score, bd = gr.rule_score(CASES[2])
    check("AOD 缺失 -> 通透度不在 breakdown", "通透度" in bd, False)
    check("能见度缺失 -> 能见度不在 breakdown", "能见度" in bd, False)
    check("其余因子仍在", "云结构" in bd and "湿度" in bd, True)
    w = {k: W7[k] for k in ("云结构", "低云遮挡", "湿度", "降水", "风速")}
    manual = sum(w[k] * bd[k] for k in w) / sum(w.values())
    check("按可用集合重归一化", score, round(manual, 1))

    # 低云缺失 -> 低云遮挡移出
    f = dict(CASES[0])
    f["cloud_low"] = None
    _, bd2 = gr.rule_score(f)
    check("低云缺失 -> 低云遮挡移出", "低云遮挡" in bd2, False)
    check("低云缺失不影响云结构", "云结构" in bd2, True)


def test_weighting():
    print("\n[4] 有剖面数据时的加权与归一化")
    f = dict(CASES[0], transect={"boundary_score": 95.0, "block_score": 100.0})
    score, bd = gr.rule_score(f)
    # 只有两个方向性因子可用 -> 基础因子缩放到 0.86
    w = {k: W7[k] * 0.86 for k in W7}
    w["剖面云边界"] = 0.08
    w["太阳方位遮挡"] = 0.06
    manual = sum(w[k] * bd[k] for k in w) / sum(w.values())
    check("两项方向性因子加权", score, round(manual, 1))
    check("breakdown 含剖面云边界", "剖面云边界" in bd, True)
    check("breakdown 含太阳方位遮挡", "太阳方位遮挡" in bd, True)
    check("breakdown 不含地形遮蔽", "地形遮蔽" in bd, False)

    # 三项齐全 -> 基础因子缩放到 0.84（与旧版 total 一致）
    f = dict(CASES[0], transect={"boundary_score": 95.0, "block_score": 100.0,
                                 "terrain_score": 82.0})
    score3, bd3 = gr.rule_score(f)
    w3 = {k: W7[k] * 0.84 for k in W7}
    w3["剖面云边界"] = 0.08
    w3["太阳方位遮挡"] = 0.06
    w3["地形遮蔽"] = 0.02
    manual3 = sum(w3[k] * bd3[k] for k in w3) / sum(w3.values())
    check("三项方向性因子加权", score3, round(manual3, 1))
    check("breakdown 含地形遮蔽", "地形遮蔽" in bd3, True)

    # 方向性因子不可用时基础权重归一回 1.0
    f = dict(CASES[0], transect={"boundary_score": None, "block_score": 100.0})
    score2, bd2 = gr.rule_score(f)
    base2 = sum(W7[k] * bd2[k] for k in W7) / sum(W7.values())
    manual2 = (0.94 * base2 + 0.06 * 100.0) / 1.0
    check("仅一项可用时按 0.94 归一", score2, round(manual2, 1))
    check("此时不含剖面云边界", "剖面云边界" in bd2, False)


def test_factor_note():
    print("\n[5] 最差因子提示")
    note = gr.factor_note({"云结构": 90, "低云遮挡": 88, "湿度": 85, "降水": 100,
                           "风速": 95, "通透度": 92, "能见度": 90,
                           "剖面云边界": 0.0, "太阳方位遮挡": 95})
    check("识别出剖面云边界", "云层边缘" in note, True)
    note = gr.factor_note({"云结构": 90, "地形遮蔽": 10.0})
    check("识别出地形遮蔽", "山体" in note, True)


def test_geometry():
    print("\n[6] 航点几何、阴影高度角与光照路径")
    lat, lon = 34.34, 108.94
    l2, n2 = solar.destination_point(lat, lon, 90, 500)
    check("正东 500km 经度", round(n2, 1), 114.4, tol=0.05)
    check("正东 500km 纬度基本不变", round(l2, 1), 34.2, tol=0.15)
    check("5km 层阴影角", round(solar.shadow_height_deg(5, lat), 2), 2.27, tol=0.01)
    check("10km 层阴影角", round(solar.shadow_height_deg(10, lat), 2), 3.21, tol=0.01)
    # 日出日落时刻的太阳高度角应恰好是 -0.833°（二分精修的意义）
    pos = solar.event_position(lat, lon, "2026-09-29T18:31", 28800)
    check("日落时刻太阳高度角", round(pos["elevation"], 2), -0.83, tol=0.02)
    # 折射常量显式化后仍等于 0.833
    check("SUNRISE_ELEVATION 常量", round(solar.SUNRISE_ELEVATION, 3), -0.833, tol=1e-6)

    # 光照几何：日落瞬间阳光与地面相切于约 185km
    lh = solar.light_horizon_km(-0.833, lat)
    check("日落时光照起点距离", round(lh, 1), 185.3, tol=0.5)
    check("相切点处光线离地高度≈0",
          round(solar.light_path_height_km(lh, -0.833, lat), 2), 0.0, tol=0.02)
    check("400km 处光线离地高度", round(solar.light_path_height_km(400, -0.833, lat), 2),
          6.74, tol=0.05)
    check("太阳在地平线上 -> 光照起点 0", solar.light_horizon_km(5.0, lat), 0.0)


def test_horizon_interpolation():
    print("\n[6b] 地形剖面环形插值")
    prof = [(0.0, 1.0), (90.0, 3.0), (180.0, 2.0), (270.0, 0.5)]
    check("0° -> 1.0", horizon.interpolate(prof, 0.0), 1.0)
    check("90° -> 3.0", horizon.interpolate(prof, 90.0), 3.0)
    check("45° 线性插值 -> 2.0", horizon.interpolate(prof, 45.0), 2.0)
    check("350° 环形回绕", round(horizon.interpolate(prof, 350.0), 3), 0.944, tol=1e-3)
    check("空剖面 -> None", horizon.interpolate([], 90.0), None)
    check("单点剖面", horizon.interpolate([(10.0, 4.0)], 200.0), 4.0)


def test_feature_missing_semantics():
    print("\n[7] 特征缺失值语义（None 而不是 0）")
    hourly = {
        "time": [f"2026-09-29T{h:02d}:00" for h in range(24)],
        "cloud_cover": [50] * 24,
        "cloud_cover_low": [None] * 24,       # 缺分层低云
        "cloud_cover_mid": [30] * 24,
        "cloud_cover_high": [20] * 24,
        "relative_humidity_2m": [55] * 24,
        "wind_speed_10m": [4] * 24,
        "precipitation": [1.0] * 6 + [0.0] * 18,   # 日出前 6 小时有雨
        "temperature_2m": [20] * 24,
        "weather_code": [1] * 24,
        "visibility": [None] * 24,
        "precipitation_probability": [None] * 24,
    }
    daily = {"time": ["2026-09-29"],
             "sunrise": ["2026-09-29T06:36"],
             "sunset": ["2026-09-29T18:31"]}
    feats = feat_mod.extract_day_features({"hourly": hourly, "daily": daily})
    f = feats[("2026-09-29", "morning")]
    check("缺分层低云 -> None", f["cloud_low"], None)
    check("缺能见度 -> None", f["visibility"], None)
    check("缺降水概率 -> None", f["precip_prob"], None)
    check("云量正常取均值", f["cloud_cover"], 50.0)
    check("窗口前 24h 降水累计", f["precip_24h"], 6.0)

    score, bd = gr.rule_score(f)
    check("低云遮挡移出加权", "低云遮挡" in bd, False)
    check("通透度移出加权", "通透度" in bd, False)
    check("能见度移出加权", "能见度" in bd, False)
    check("云结构仍在", "云结构" in bd, True)


def test_model_vector_and_gate():
    print("\n[8] 模型特征向量与真实标签闸门")
    from src.model import FEATURE_NAMES, GlowModel, feature_vector

    f = {"cloud_cover": 50, "cloud_low": 10, "cloud_mid": 30, "cloud_high": 20,
         "humidity": 55, "wind": 4, "precip": 0, "temp": 20, "day_of_year": 100,
         "visibility": 18000, "aod": 0.2}
    check("特征维度", len(feature_vector(f)), len(FEATURE_NAMES))
    check("特征中不含 rule_score", "rule_score" in FEATURE_NAMES, False)
    # None 安全：分层云量缺失也不能崩
    f2 = dict(f, cloud_low=None)
    vec = feature_vector(f2)
    check("None 安全不抛异常", len(vec), len(FEATURE_NAMES))
    check("缺 visibility 用中性值", feature_vector({})[-2], 10000.0)
    check("缺 aod 用中性值", feature_vector({})[-1], 0.3)

    # 闸门：无真实观测标注时不得训练
    import src.train as tr
    rows = [
        {"date": f"2026-01-{i % 28 + 1:02d}", "window": "evening",
         "glow": i % 2, "source": "weak-positive" if i % 2 else "weak-negative",
         "cloud_cover": 50, "cloud_low": 10, "cloud_mid": 30, "cloud_high": 20,
         "humidity": 55, "wind": 4, "precip": 0, "temp": 20, "day_of_year": 100,
         "visibility": 15000, "aod": 0.3, "rule_score": 70 if i % 2 else 30}
        for i in range(40)
    ]
    stats = {"total": 40, "posts": 0, "post_rows": 0, "positive": 20, "negative": 20}
    cfg = {"model": {"type": "logistic", "bootstrap": True, "min_samples": 30,
                     "min_real_labels": 30, "holdout_ratio": 0.3,
                     "model_path": os.path.join(tempfile.gettempdir(), "never.joblib")}}
    with mock.patch.object(tr, "build_training_rows", return_value=(rows, stats)):
        res = tr.run_training(cfg)
    check("无真实标注 -> 不训练", res["trained"], False)
    check("原因说明真实观测不足", "真实观测标注" in res.get("reason", ""), True)

    # 旧维度模型（含 rule_score 的 13 维）必须被弃用
    m = GlowModel("logistic")
    m.model = _FakePipe(13)
    path = os.path.join(tempfile.gettempdir(), "stale_model_test.joblib")
    joblib.dump(m, path)
    check("13 维旧模型被弃用", GlowModel.load(path), None)
    m.model = _FakePipe(len(FEATURE_NAMES))
    joblib.dump(m, path)
    check("12 维模型可加载", GlowModel.load(path) is not None, True)
    try:
        os.remove(path)
    except OSError:
        pass


def test_crosscheck_layer_check():
    print("\n[8b] 交叉验证的分层云量校验")
    ok = {"time": [], "cloud_cover": [10], "cloud_cover_low": [1],
          "cloud_cover_mid": [1], "cloud_cover_high": [1]}
    bad = {"time": [], "cloud_cover": [10], "cloud_cover_low": [None],
           "cloud_cover_mid": [1], "cloud_cover_high": [1]}
    missing = {"time": [], "cloud_cover": [10]}
    check("分层齐全 -> 有效", crosscheck._valid_hourly(ok), True)
    check("分层低云全空 -> 无效", crosscheck._valid_hourly(bad), False)
    check("完全缺分层字段 -> 无效", crosscheck._valid_hourly(missing), False)


def test_rate_limit_breaker():
    """429 必须立刻熔断，且冷却期内不再发出任何网络请求。

    这是防止"被限流后重试放大请求量、把分钟级限流升级成整点封禁"的关键保护。
    """
    print("\n[9] 限流熔断")
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


def test_history_endpoint():
    print("\n[10] 训练侧数据源已切到 Historical Forecast API")
    check("不再使用 ERA5 archive-api",
          "archive-api" in weather.HISTORICAL_FORECAST_URL, False)
    check("指向 historical-forecast-api",
          "historical-forecast-api" in weather.HISTORICAL_FORECAST_URL, True)


def main():
    for fn in (test_boundary_curve, test_block_curve, test_terrain_curve,
               test_aod_typing, test_precip_timing, test_degradation_parity,
               test_missing_factor_removed, test_weighting, test_factor_note,
               test_geometry, test_horizon_interpolation,
               test_feature_missing_semantics, test_model_vector_and_gate,
               test_crosscheck_layer_check, test_rate_limit_breaker,
               test_history_endpoint):
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
