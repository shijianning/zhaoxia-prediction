"""全国朝霞/晚霞预测地图模块。

对 data/cities.json 中的全国城市，逐城拉取 Open-Meteo 预报 + 气溶胶(AOD)，
用气象规则引擎算出今明两天朝霞/晚霞评分，并渲染成一张中国地图网页。

坐标体系：
- Open-Meteo / cities.json 使用 WGS-84；
- 腾讯地图 GL JS 使用 GCJ-02（火星坐标），展示前需转换。

合规说明：
- 地图使用腾讯地图 GL JS 的代理模式（前端不携带 key，经 WorkBuddy 本地代理），
  模板中 __WB_HTTP_PORT__ / __WB_TMAP_SECRET__ 为运行时占位符，必须原样保留。
"""
import copy
import datetime as dt
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import features as feat_mod
from . import transect
from . import weather
from .glow_rules import grade_of, rule_score, vividness_of

CITIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cities.json",
)

# 出界判断（用于 WGS-84 -> GCJ-02）
_PI = math.pi


def _out_of_china(lat, lon):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def wgs84_to_gcj02(lat, lon):
    """WGS-84 转 GCJ-02（火星坐标）。境外坐标原样返回。"""
    if _out_of_china(lat, lon):
        return lat, lon
    a = 6378245.0
    ee = 0.00669342162296594323

    def _trans_lat(x, y):
        ret = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y
               + 0.1 * x * y + 0.2 * math.sqrt(abs(x)))
        ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * _PI) + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
        return ret

    def _trans_lon(x, y):
        ret = (300.0 + x + 2.0 * y + 0.1 * x * x
               + 0.1 * x * y + 0.1 * math.sqrt(abs(x)))
        ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
        return ret

    dlat = _trans_lat(lon - 105.0, lat - 35.0)
    dlon = _trans_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * _PI)
    return lat + dlat, lon + dlon


def load_cities(path=None):
    """加载城市列表，返回 [{name, lat, lon, major}]。"""
    path = path or CITIES_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cities", [])


def _city_cfg(base_cfg, name, lat, lon, days):
    cfg = copy.deepcopy(base_cfg)
    cfg["city"] = {
        "name": name,
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Shanghai",
    }
    cfg["forecast"]["days"] = days
    return cfg


def predict_city(base_cfg, name, lat, lon, days=2):
    """预测单个城市，返回 {date: {morning: info, evening: info}}。

    info 字段：score(0-100) / vivid(鲜艳度) / grade(评级)。
    全国地图使用「气象规则引擎」评分（与地点无关），不套用仅针对西安训练的 ML 模型。
    """
    cfg = _city_cfg(base_cfg, name, lat, lon, days)
    data = weather.get_forecast(cfg)
    day_feats = feat_mod.extract_day_features(data)
    daily_wx = feat_mod.extract_daily_weather(data)

    try:
        aq = weather.get_air_quality(cfg)
        feat_mod.add_air_quality(day_feats, aq)
    except Exception:
        pass

    # 太阳方位剖面（方向性评分）：每城多一次多坐标请求，失败/关闭时自动降级
    if transect.enabled(base_cfg, "national"):
        try:
            tr_check = transect.compute_transect(cfg, data)
        except Exception:
            tr_check = {}
        for key, f in day_feats.items():
            tr = tr_check.get(key)
            if tr:
                f["transect"] = tr

    result = {}
    for key, f in day_feats.items():
        date, window = key
        score, _ = rule_score(f)
        info = {
            "score": round(score, 0),
            "vivid": vividness_of(score),
            "grade": grade_of(score),
        }
        # 方向性剖面摘要，供地图气泡展示（失败/未启用时不带该字段）
        tr = f.get("transect")
        if tr:
            bits = []
            az = tr.get("azimuth")
            if az is not None:
                bits.append(f"太阳方位 {az:.0f}°")
            km = tr.get("boundary_km")
            bits.append(f"云幕边缘 {km:.0f}km" if km is not None else "太阳向无高云")
            near = tr.get("near_low_cloud")
            if near is not None:
                bits.append(f"近场低云 {near:.0f}%")
            info["dir"] = " · ".join(bits)
        result.setdefault(date, {})[window] = info
    # 每日天气概览（日出日落/温度/降雨），供地图气泡展示
    for date, wx in daily_wx.items():
        result.setdefault(date, {})["daily"] = wx
    return result


def run_national(cfg, cities, days=2, workers=8, progress=None, retries=2):
    """并行预测所有城市，返回 (results, failures)。

    results: {name: {date: {morning/evening: info}}}
    failures: {name: error_message}
    并发请求偶发 SSL 断连 / 代理 502，失败城市会自动退避重试（retries 次）。
    """
    results = {}
    failures = {}
    pending = list(cities)
    done_base = 0
    for attempt in range(retries + 1):
        if not pending:
            break
        if weather.in_cooldown():
            # 已触发 Open-Meteo 限流：冷却期内重试只会空转（请求会被直接拒绝），
            # 立刻放弃剩余城市，把已成功的部分先落盘，避免整轮白跑。
            break
        res, fail = _run_batch(cfg, pending, days, workers,
                               progress=progress, start_done=done_base,
                               is_retry=attempt > 0)
        results.update(res)
        done_base += len(res)
        pending = [c for c in pending if c["name"] in fail]
        failures = fail
        if pending and attempt < retries and not weather.in_cooldown():
            time.sleep(2.5)
    return results, failures


def _run_batch(cfg, cities, days, workers, progress, start_done, is_retry):
    results = {}
    failures = {}
    done = start_done
    total = start_done + len(cities)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(predict_city, cfg, c["name"], c["lat"], c["lon"], days): c
            for c in cities
        }
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                results[c["name"]] = fut.result()
            except Exception as exc:
                failures[c["name"]] = str(exc)
            if progress:
                tag = "重试" if is_retry else ""
                progress(done, total, c["name"], tag)
    return results, failures


def _score_color(score):
    """分数 -> 地图散点颜色。"""
    if score >= 80:
        return "#dc2626"   # 深红：优质大烧/世纪大烧
    if score >= 70:
        return "#f97316"   # 橙：大烧
    if score >= 60:
        return "#fbbf24"   # 琥珀：中烧
    if score >= 45:
        return "#facc15"   # 黄：小烧
    return "#94a3b8"       # 灰：微烧/平淡


def _tmap_script(tmap_key):
    """生成腾讯地图 SDK 加载片段。

    - 有 key：标准模式（部署到自有服务器用，前端携带用户自己的 key）。
    - 无 key：代理模式（WorkBuddy 本地环境，占位符必须原样保留，运行时由本地代理注入）。
    """
    if tmap_key:
        return (
            '<script src="https://map.qq.com/api/gljs?v=1.exp&key='
            + html_escape(tmap_key)
            + '"></script>'
        )
    return (
        '<!-- 代理模式：先配置再加载 SDK，前端不携带 key -->\n'
        '<script type="text/javascript">\n'
        '  window._TMapSecurityConfig = {\n'
        "    serviceHost: 'http://127.0.0.1:__WB_HTTP_PORT__/_TMapService/_wbt/__WB_TMAP_SECRET__',\n"
        '  };\n'
        '</script>\n'
        '<!-- 从官方 CDN 加载 SDK，不带 key 参数 -->\n'
        '<script src="https://map.qq.com/api/gljs?v=1.exp"></script>'
    )


def render_national_map(cfg, cities, results, failures, days=2):
    """渲染全国预测地图 HTML 字符串。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # 组装前端数据：仅成功城市；坐标转 GCJ-02
    rows = []
    for c in cities:
        name = c["name"]
        if name not in results:
            continue
        glat, glon = wgs84_to_gcj02(c["lat"], c["lon"])
        rows.append({
            "name": name,
            "lat": round(glat, 5),
            "lon": round(glon, 5),
            "major": bool(c.get("major", False)),
            "days": results[name],
        })

    payload = json.dumps(rows, ensure_ascii=False)
    n_ok = len(rows)
    n_fail = len(failures)

    html = _MAP_TEMPLATE
    html = html.replace("__CITIES_JSON__", payload)
    html = html.replace("__NOW__", now)
    html = html.replace("__N_OK__", str(n_ok))
    html = html.replace("__N_FAIL__", str(n_fail))
    html = html.replace("__FAIL_LIST__", _fail_list_html(failures))
    tmap_key = (cfg.get("map") or {}).get("tmap_key", "")
    html = html.replace("__TMAP_SCRIPT__", _tmap_script(tmap_key))
    return html


def _fail_list_html(failures):
    if not failures:
        return ""
    items = "".join(
        f'<span class="fail-chip">{html_escape(name)}'
        f'<i title="{html_escape(msg)}"> ⚠</i></span>'
        for name, msg in list(failures.items())
    )
    return f'<div class="fail-note">未取到数据（{len(failures)}）：{items}</div>'


def html_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def save_national_map(cfg, cities, results, failures, days=2):
    """渲染并写入 output/national_map.html，返回文件路径。"""
    html_str = render_national_map(cfg, cities, results, failures, days=days)
    out_path = os.path.join(cfg["output"]["dir"], "national_map.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return out_path


def save_cache(cfg, results, failures):
    """把预测结果落盘缓存，便于 --render-only 快速重渲染。"""
    path = os.path.join(cfg["output"]["dir"], "national_map_cache.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "failures": failures}, f, ensure_ascii=False)
    return path


def load_cache(cfg):
    """读取缓存，返回 (results, failures)；无缓存时返回 (None, None)。"""
    path = os.path.join(cfg["output"]["dir"], "national_map_cache.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("results"), d.get("failures")


_MAP_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全国朝霞晚霞预测地图</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif; }
  #wrap { display: flex; flex-direction: column; height: 100vh; background: #f8fafc; }
  header { padding: 14px 18px; background: linear-gradient(135deg,#fb7185,#f97316,#8b5cf6); color: #fff; }
  header h1 { margin: 0; font-size: 20px; font-weight: 800; }
  header .sub { font-size: 12px; opacity: .92; margin-top: 4px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 10px 18px; background: #fff; border-bottom: 1px solid #e2e8f0; }
  .toolbar .group-label { font-size: 12px; color: #64748b; margin-right: 2px; }
  .btn { border: 1px solid #e2e8f0; background: #fff; color: #334155; border-radius: 999px; padding: 6px 14px; font-size: 13px; cursor: pointer; transition: all .15s; }
  .btn:hover { border-color: #f97316; color: #f97316; }
  .btn.active { background: #f97316; border-color: #f97316; color: #fff; font-weight: 700; }
  #map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
  .legend { position: absolute; z-index: 10; left: 18px; bottom: 26px; background: rgba(255,255,255,.96); border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); font-size: 12px; color: #334155; }
  .legend .lg-title { font-weight: 700; margin-bottom: 6px; }
  .legend .lg-row { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .legend .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; border: 1.5px solid #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.08); }
  .footer { position: absolute; z-index: 10; right: 12px; bottom: 8px; font-size: 11px; color: #94a3b8; background: rgba(255,255,255,.85); padding: 4px 8px; border-radius: 8px; }
  .fail-note { position: absolute; z-index: 10; top: 8px; left: 18px; right: 18px; font-size: 12px; color: #b45309; }
  .fail-chip { display: inline-block; background: rgba(255,247,237,.95); border: 1px solid #fcd34d; border-radius: 6px; padding: 1px 6px; margin: 2px; }
  .iw { min-width: 210px; max-width: 260px; font-size: 13px; color: #1e293b; }
  .iw h3 { margin: 0 0 6px; font-size: 15px; }
  .iw .row { display: flex; justify-content: space-between; padding: 4px 0; border-top: 1px dashed #e2e8f0; }
  .iw .row .k { color: #64748b; }
  .iw .val { font-weight: 700; }
</style>
<!-- 地图 SDK 加载（占位符 __TMAP_SCRIPT__ 在渲染时替换为 key 模式或代理模式） -->
__TMAP_SCRIPT__
</head>
<body>
<div id="wrap">
  <header>
    <h1>全国朝霞 · 晚霞预测地图</h1>
    <div class="sub">更新于 __NOW__ · 覆盖 __N_OK__ 城（__N_FAIL__ 城暂无数据）· 数据源 Open-Meteo · 气象规则引擎评分</div>
  </header>
  <div class="toolbar">
    <span class="group-label">视图</span>
    <button class="btn view-btn" data-view="dot">散点</button>
    <button class="btn view-btn" data-view="heat">热力</button>
    <span class="group-label" style="margin-left:8px;">日期</span>
    <button class="btn date-btn" data-date="0">今天</button>
    <button class="btn date-btn" data-date="1">明天</button>
    <span class="group-label" style="margin-left:8px;">时段</span>
    <button class="btn win-btn" data-win="evening">晚霞 🌇</button>
    <button class="btn win-btn" data-win="morning">朝霞 🌅</button>
  </div>
  <div style="position:relative;flex:1;min-height:0;">
    <div id="map"></div>
    __FAIL_LIST__
    <div class="legend">
      <div class="lg-title">出霞评分（0–100）</div>
      <div class="lg-row"><span class="dot" style="background:#dc2626"></span> ≥80 优质大烧 / 世纪大烧</div>
      <div class="lg-row"><span class="dot" style="background:#f97316"></span> 70–79 大烧</div>
      <div class="lg-row"><span class="dot" style="background:#fbbf24"></span> 60–69 中烧</div>
      <div class="lg-row"><span class="dot" style="background:#facc15"></span> 45–59 小烧</div>
      <div class="lg-row"><span class="dot" style="background:#94a3b8"></span> &lt;45 微烧 / 平淡</div>
    </div>
    <div class="footer">地图：腾讯地图 · 坐标 GCJ-02 · 仅供摄影参考</div>
  </div>
</div>
<script>
  const CITIES = __CITIES_JSON__;

  const DATES = Object.keys(CITIES[0].days).sort();
  let curDateIdx = 0;
  let curWin = 'evening';

  function scoreColor(s) {
    if (s >= 80) return '#dc2626';
    if (s >= 70) return '#f97316';
    if (s >= 60) return '#fbbf24';
    if (s >= 45) return '#facc15';
    return '#94a3b8';
  }

  function markerSrc(score, size) {
    const r = Math.round(size / 2);
    const c = scoreColor(score);
    const textColor = score >= 70 ? '#ffffff' : '#3f3f46';
    const svg =
      '<svg xmlns="http://www.w3.org/2000/svg" width="' + size + '" height="' + size + '">' +
      '<circle cx="' + r + '" cy="' + r + '" r="' + (r - 2) + '" fill="' + c + '" stroke="#ffffff" stroke-width="2"/>' +
      '<text x="' + r + '" y="' + (r + 4) + '" font-size="' + (size >= 36 ? 12 : 10) + '" fill="' + textColor +
      '" text-anchor="middle" font-family="sans-serif" font-weight="bold">' + score + '</text>' +
      '</svg>';
    return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  }

  const map = new TMap.Map('map', {
    center: new TMap.LatLng(35.5, 105.5),
    zoom: 4.2,
  });

  // 为每个可能分数(0-100)预建两套样式（普通 / 省会）
  const styles = {};
  const majorStyles = {};
  for (let s = 0; s <= 100; s++) {
    styles['s' + s] = new TMap.MarkerStyle({ width: 30, height: 30, anchor: { x: 15, y: 15 }, src: markerSrc(s, 30) });
    majorStyles['m' + s] = new TMap.MarkerStyle({ width: 38, height: 38, anchor: { x: 19, y: 19 }, src: markerSrc(s, 38) });
  }

  const allMarkerStyles = Object.assign({}, styles, majorStyles);
  const markers = new TMap.MultiMarker({ map: map, styles: allMarkerStyles, geometries: [] });
  const labels = new TMap.MultiLabel({
    map: map,
    styles: {
      name: new TMap.LabelStyle({
        color: '#334155', size: 11, offset: { x: 0, y: 22 },
        angle: 0, alignment: 'center', verticalAlignment: 'middle',
      }),
    },
    geometries: [],
  });

  // 热力图图层（融合 weather-sunset-predictor 的火烧云热力图能力）
  // count = 出霞评分(0-100)，暖色=高、灰蓝=低；与散点同一配色体系
  const heat = new TMap.visualization.Heat({
    radius: 38,
    gradient: {
      0.0: 'rgba(148,163,184,0)',
      0.35: '#94a3b8',
      0.5: '#facc15',
      0.65: '#fbbf24',
      0.8: '#f97316',
      1.0: '#dc2626',
    },
    opacity: 0.85,
    min: 0,
    max: 100,
  });
  heat.addTo(map);
  heat.hide();
  let curView = 'dot';

  function buildHeatData() {
    const date = DATES[curDateIdx];
    const data = [];
    CITIES.forEach(function (c) {
      const day = c.days[date];
      if (!day) return;
      const win = day[curWin];
      if (!win) return;
      data.push({ lat: c.lat, lng: c.lon, count: Math.round(win.score) });
    });
    return data;
  }

  function render() {
    const date = DATES[curDateIdx];
    const geoms = [];
    const labelGeoms = [];
    CITIES.forEach(function (c) {
      const day = c.days[date];
      if (!day) return;
      const win = day[curWin];
      if (!win) return;
      const s = Math.round(win.score);
      const styleId = (c.major ? 'm' : 's') + s;
      const p = new TMap.LatLng(c.lat, c.lon);
      geoms.push({ id: c.name, styleId: styleId, position: p, properties: { name: c.name } });
      if (c.major) {
        labelGeoms.push({ id: 'lb-' + c.name, styleId: 'name', position: p, content: c.name });
      }
    });
    markers.setGeometries(geoms);
    labels.setGeometries(labelGeoms);
  }

  // 统一刷新：根据当前视图切换「散点」或「热力」
  function refresh() {
    if (curView === 'heat') {
      heat.setData(buildHeatData());
      markers.hide();
      labels.hide();
      heat.show();
    } else {
      heat.hide();
      markers.show();
      labels.show();
      render();
    }
  }

  const infoWindow = new TMap.InfoWindow({
    map: map,
    enableCustom: true,
    position: new TMap.LatLng(35.5, 105.5),
    offset: { x: 0, y: -20 },
  });
  infoWindow.close();

  function contentHtml(c) {
    const d = c.days[DATES[curDateIdx]];
    const m = d && d.morning ? d.morning : null;
    const e = d && d.evening ? d.evening : null;
    const fmt = function (w) {
      if (!w) return '<span style="color:#94a3b8">暂无</span>';
      let s = w.score + ' 分 · ' + w.vivid;
      if (w.dir) s += '<div style="font-size:10px;color:#94a3b8;margin-top:1px;">🧭 ' + w.dir + '</div>';
      return s;
    };
    const dw = d && d.daily ? d.daily : null;
    let wxLine = '';
    if (dw) {
      wxLine = '<div class="row"><span class="k">天气</span><span class="val">' + dailyLine(dw) + '</span></div>';
    }
    return '<div class="iw"><h3>' + c.name + (c.major ? ' <span style="font-size:11px;color:#f97316;">省会</span>' : '') + '</h3>' +
      '<div style="font-size:11px;color:#94a3b8;">' + DATES[curDateIdx] + '</div>' +
      '<div class="row"><span class="k">朝霞 🌅</span><span class="val">' + fmt(m) + '</span></div>' +
      '<div class="row"><span class="k">晚霞 🌇</span><span class="val">' + fmt(e) + '</span></div>' +
      wxLine +
      '</div>';
  }

  // WMO 天气码 -> 中文+emoji（与单城报告一致）
  const WX = {0:'☀️晴',1:'🌤️大致晴',2:'⛅多云',3:'☁️阴',45:'🌫️雾',48:'🌫️雾凇',
    51:'🌦️小毛毛雨',53:'🌦️毛毛雨',55:'🌧️大毛毛雨',61:'🌦️小雨',63:'🌧️中雨',65:'🌧️大雨',
    66:'🌧️冻雨',67:'🌧️强冻雨',71:'🌨️小雪',73:'🌨️中雪',75:'❄️大雪',77:'🌨️米雪',
    80:'🌦️阵雨',81:'🌧️强阵雨',82:'⛈️暴雨',85:'🌨️阵雪',86:'❄️强阵雪',
    95:'⛈️雷暴',96:'⛈️雷暴冰雹',99:'⛈️强雷暴冰雹'};
  function wxText(code) { return WX[code] || '🌡️未知'; }
  function hhmm(iso) { return iso ? String(iso).slice(11, 16) : '--:--'; }
  function num(v, unit) { return (v === null || v === undefined) ? '--' : (Math.round(v * 10) / 10) + unit; }
  function dailyLine(dw) {
    let s = wxText(dw.code);
    if (dw.tmax !== null && dw.tmin !== null && dw.tmax !== undefined && dw.tmin !== undefined) {
      s += ' ' + Math.round(dw.tmin) + '~' + Math.round(dw.tmax) + '℃';
    }
    s += ' · 🌅' + hhmm(dw.sunrise) + ' 🌇' + hhmm(dw.sunset);
    if (dw.precip_sum !== null && dw.precip_sum !== undefined && dw.precip_sum > 0) {
      s += ' · 💧' + num(dw.precip_sum, 'mm');
    }
    return s;
  }

  markers.on('click', function (e) {
    const geo = e.geometry;
    if (!geo || !geo.properties) return;
    const name = geo.properties.name;
    const c = CITIES.find(function (x) { return x.name === name; });
    if (!c) return;
    infoWindow.setPosition(new TMap.LatLng(c.lat, c.lon));
    infoWindow.setContent(contentHtml(c));
    infoWindow.open();
  });

  function bindButtons() {
    document.querySelectorAll('.view-btn').forEach(function (b) {
      b.addEventListener('click', function () {
        curView = b.getAttribute('data-view');
        document.querySelectorAll('.view-btn').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        refresh();
      });
    });
    document.querySelectorAll('.date-btn').forEach(function (b) {
      b.addEventListener('click', function () {
        curDateIdx = parseInt(b.getAttribute('data-date'), 10);
        document.querySelectorAll('.date-btn').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        refresh();
      });
    });
    document.querySelectorAll('.win-btn').forEach(function (b) {
      b.addEventListener('click', function () {
        curWin = b.getAttribute('data-win');
        document.querySelectorAll('.win-btn').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        refresh();
      });
    });
  }

  bindButtons();
  document.querySelector('.view-btn[data-view="dot"]').classList.add('active');
  document.querySelector('.date-btn[data-date="0"]').classList.add('active');
  document.querySelector('.win-btn[data-win="evening"]').classList.add('active');
  refresh();
</script>
</body>
</html>
"""
