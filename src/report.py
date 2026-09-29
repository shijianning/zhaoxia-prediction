"""HTML 报告生成模块。

生成自包含的网页报告，展示未来几天朝霞/晚霞评分与因素拆解。
"""
import datetime as dt
import html
import json

from .crosscheck import MODEL_LABEL
from .glow_rules import chroma_index, factor_note, vividness_of


def _bar_color(score):
    """根据评分返回进度条颜色（暖色=出霞概率高）。"""
    if score >= 75:
        return "#f97316"
    if score >= 60:
        return "#fbbf24"
    if score >= 45:
        return "#94a3b8"
    return "#cbd5e1"


def _grade_badge(grade, score):
    colors = {
        "优秀": ("#f97316", "#fff7ed"),
        "良好": ("#f59e0b", "#fffbeb"),
        "一般": ("#64748b", "#f1f5f9"),
        "平淡": ("#94a3b8", "#f8fafc"),
    }
    c, bg = colors.get(grade, ("#64748b", "#f1f5f9"))
    return f'<span style="background:{bg};color:{c};padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;">{grade}</span>'


def _xcheck_line(xc):
    """渲染多模型交叉验证信息。"""
    if not xc:
        return ""
    parts = " / ".join(
        f"{MODEL_LABEL.get(m, m)} {s:.0f}" for m, s in xc["scores"].items()
    )
    color = {"高": "#16a34a", "中": "#f59e0b", "低": "#ef4444"}.get(
        xc["confidence"], "#64748b"
    )
    agree = "结论一致" if xc["agree"] else "存在分歧"
    return (
        f'<div style="margin-top:8px;font-size:11px;color:#64748b;'
        f'background:#f8fafc;border-radius:8px;padding:6px 8px;line-height:1.6;">'
        f'🔍 交叉验证：{html.escape(parts)} · 分差 {xc["spread"]:.0f} · '
        f'置信度 <span style="color:{color};font-weight:700;">{xc["confidence"]}</span>'
        f'（{agree}）</div>'
    )


def _geovisearth_line(gv):
    """渲染星图云官方火烧云预报的第三方对比信息。"""
    if not gv:
        return ""
    grade = gv.get("grade", "—")
    score = gv.get("score")
    score_str = f' {score:.0f}分' if score is not None else ""
    return (
        f'<div style="margin-top:8px;font-size:11px;color:#7c3aed;'
        f'background:#f5f3ff;border-radius:8px;padding:6px 8px;line-height:1.6;">'
        f'🌐 星图云官方预报：{html.escape(grade)}{score_str}</div>'
    )


def _transect_panel(tr):
    """太阳方位剖面小图。

    把"从观测点朝太阳方向 0~500km 这条线上有什么云"画出来 —— 这是本项目
    相对"只看头顶云量"的关键升级：朝霞晚霞的光来自太阳方位，承光的幕布和
    它的边缘位置都在几十到几百公里之外。

    用纯内联 SVG 绘制（无 JS、无外部请求），柱子越高云量越大：
    浅蓝 = 高云（承光的幕布），灰色 = 低云（会挡光）。
    """
    if not tr:
        return ""
    # 幕布用中云+高云（与"剖面云边界"因子口径一致），低云单独画
    hcc = tr.get("ray_canopy") or tr.get("ray_hcc") or []
    lcc = tr.get("ray_lcc") or []
    dists = tr.get("distances_km") or []
    n = len(hcc) or len(lcc)
    if n == 0:
        return ""

    pad_l, pad_r, top, bottom = 26.0, 26.0, 10.0, 52.0
    plot_w = 320.0 - pad_l - pad_r
    slot = plot_w / n
    bar_w = max(2.0, slot - 2.0)
    span = bottom - top

    def bars(series, color, opacity):
        if not series:
            return ""
        out = []
        for i, v in enumerate(series):
            if v is None:
                continue
            h = max(0.6, float(v) / 100.0 * span)
            x = pad_l + i * slot + (slot - bar_w) / 2.0
            out.append(
                f'<rect x="{x:.1f}" y="{bottom - h:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{color}" opacity="{opacity}" rx="1"/>'
            )
        return "".join(out)

    # 云层边缘标记
    marker = ""
    edge_km = tr.get("boundary_km")
    if edge_km is not None and dists:
        max_d = max(dists) or 1
        x = pad_l + min(float(edge_km), max_d) / max_d * plot_w
        marker = (
            f'<line x1="{x:.1f}" y1="{top - 3}" x2="{x:.1f}" y2="{bottom}" '
            f'stroke="#ef4444" stroke-width="1.4" stroke-dasharray="3 2"/>'
            f'<text x="{x:.1f}" y="{top - 5}" font-size="8" fill="#ef4444" '
            f'text-anchor="middle">边缘</text>'
        )

    axis = (
        f'<line x1="{pad_l}" y1="{bottom}" x2="{pad_l + plot_w}" y2="{bottom}" '
        f'stroke="#cbd5e1" stroke-width="1"/>'
        f'<text x="{pad_l}" y="{bottom + 9}" font-size="8" fill="#94a3b8" '
        f'text-anchor="middle">0</text>'
        f'<text x="{pad_l + plot_w * 0.4:.0f}" y="{bottom + 9}" font-size="8" '
        f'fill="#94a3b8" text-anchor="middle">200</text>'
        f'<text x="{pad_l + plot_w * 0.8:.0f}" y="{bottom + 9}" font-size="8" '
        f'fill="#94a3b8" text-anchor="middle">400</text>'
        f'<text x="{pad_l + plot_w:.0f}" y="{bottom + 9}" font-size="8" '
        f'fill="#94a3b8" text-anchor="middle">500km</text>'
    )

    az = tr.get("azimuth")
    elev = tr.get("sun_elev")
    shadow = tr.get("shadow_deg")
    near = tr.get("near_low_cloud")
    terrain = tr.get("terrain_deg")
    light_h = tr.get("light_horizon_km")

    facts = []
    if az is not None:
        facts.append(f"太阳方位 {az:.0f}°")
    if elev is not None:
        facts.append(f"高度角 {elev:.2f}°")
    if edge_km is not None:
        facts.append(f"云层边缘 {edge_km:.0f}km")
    else:
        facts.append("太阳方向无中高云幕")
    if near is not None:
        facts.append(f"近场低云 {near:.0f}%")
    if terrain is not None and terrain > 0.05:
        facts.append(f"地形仰角 {terrain:.1f}°")
    if shadow is not None:
        facts.append(f"3km 层阴影角 {shadow:.2f}°")

    # 光照几何提示：阳光与地面相切的距离 —— 比这更近的地方光线仍在地面之下
    light_note = ""
    if light_h is not None and light_h > 0:
        light_note = (
            f'<div style="font-size:10px;color:#94a3b8;line-height:1.5;margin-top:2px;">'
            f'💡 此刻阳光与地面相切于 <strong>{light_h:.0f}km</strong> —— '
            f'比这更近处光线仍在地面之下，那里的云底照不到光</div>'
        )

    svg = (
        f'<svg viewBox="0 0 320 64" width="100%" height="72" '
        f'style="display:block;overflow:visible;" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{bars(lcc, "#cbd5e1", 0.9)}'
        f'{bars(hcc, "#60a5fa", 0.85)}'
        f'{marker}{axis}</svg>'
    )

    return (
        f'<div style="margin-top:10px;padding:10px;background:#f8fafc;'
        f'border:1px solid #e2e8f0;border-radius:10px;">'
        f'<div style="font-size:11px;font-weight:700;color:#334155;margin-bottom:4px;">'
        f'🧭 太阳方位剖面（0→500km，朝日出/日落方向）</div>'
        f'<div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">'
        f'<span style="color:#60a5fa;">■</span> 中高云（承接阳光的幕布）&nbsp;'
        f'<span style="color:#cbd5e1;">■</span> 低云（遮挡低角度光线）</div>'
        f'{svg}'
        f'<div style="font-size:10px;color:#64748b;line-height:1.6;">'
        f'{html.escape(" · ".join(facts))}</div>'
        f'{light_note}'
        f'</div>'
    )


def _window_block(title, emoji, f):
    score = f["final"]
    note = factor_note(f["breakdown"])
    bar = _bar_color(score)
    vivid = vividness_of(score)
    chroma = f.get("chroma")
    if chroma is None:
        chroma = chroma_index(score)
    aod = f.get("aod")
    aod_str = f'气溶胶AOD {aod:.2f}' if aod is not None else "AOD 暂无"
    vis = f.get("visibility")
    vis_str = f'能见度 {vis / 1000:.1f}km' if vis is not None else "能见度 暂无"
    factors = " · ".join([
        f'云量 {f["cloud_cover"]:.0f}%',
        f'湿度 {f["humidity"]:.0f}%',
        f'风 {f["wind"]:.1f}m/s',
        f'降水 {f["precip"]:.1f}mm',
        aod_str,
        vis_str,
    ])
    return f'''
    <div style="padding:14px 16px;border-top:1px solid #f1f5f9;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-weight:700;font-size:15px;color:#1e293b;">{emoji} {title}</div>
        <div style="display:flex;gap:6px;align-items:center;">
          <span style="font-size:12px;font-weight:700;color:#c2410c;">{vivid}</span>
          <span style="font-size:12px;font-weight:700;color:#7c3aed;">🔥 {chroma}/10</span>
          {_grade_badge(f["grade"], score)}
        </div>
      </div>
      <div style="display:flex;align-items:baseline;gap:8px;margin:6px 0 8px;">
        <span style="font-size:30px;font-weight:800;color:{bar};">{score:.0f}</span>
        <span style="font-size:13px;color:#64748b;">/ 100</span>
      </div>
      <div style="height:8px;background:#f1f5f9;border-radius:999px;overflow:hidden;">
        <div style="width:{score:.0f}%;height:100%;background:linear-gradient(90deg,{bar},{bar});border-radius:999px;"></div>
      </div>
      <div style="margin-top:10px;font-size:12px;color:#475569;line-height:1.6;">{html.escape(note)}</div>
      <div style="margin-top:6px;font-size:11px;color:#94a3b8;">{html.escape(factors)}</div>
      {_transect_panel(f.get("transect"))}
      {_xcheck_line(f.get("xcheck"))}
      {_geovisearth_line(f.get("geovisearth"))}
    </div>'''


_WEATHER_CODE = {
    0: ("晴", "☀️"),
    1: ("晴间多云", "🌤️"),
    2: ("多云", "⛅"),
    3: ("阴", "☁️"),
    45: ("雾", "🌫️"),
    48: ("雾凇", "🌫️"),
    51: ("毛毛雨", "🌦️"),
    53: ("毛毛雨", "🌦️"),
    55: ("毛毛雨", "🌦️"),
    56: ("冻毛毛雨", "🌧️"),
    57: ("冻毛毛雨", "🌧️"),
    61: ("小雨", "🌧️"),
    63: ("中雨", "🌧️"),
    65: ("大雨", "🌧️"),
    66: ("冻雨", "🌧️"),
    67: ("冻雨", "🌧️"),
    71: ("小雪", "🌨️"),
    73: ("中雪", "🌨️"),
    75: ("大雪", "❄️"),
    77: ("雪粒", "❄️"),
    80: ("阵雨", "🌧️"),
    81: ("阵雨", "🌧️"),
    82: ("强阵雨", "⛈️"),
    85: ("阵雪", "🌨️"),
    86: ("强阵雪", "❄️"),
    95: ("雷暴", "⛈️"),
    96: ("雷暴伴冰雹", "⛈️"),
    99: ("强雷暴伴冰雹", "⛈️"),
}


def _wx_desc(code):
    """WMO 天气代码 -> (中文描述, emoji)。"""
    if code is None:
        return "—", "🌫️"
    return _WEATHER_CODE.get(int(code), ("—", "🌫️"))


def _hhmm(iso):
    """ISO 时间字符串 'YYYY-MM-DDTHH:MM' -> 'HH:MM'。"""
    if not iso:
        return ""
    return iso[11:16]


def _daily_weather_line(daily):
    """渲染每日天气概览条（天气现象 / 温度 / 日出日落 / 降水）。"""
    if not daily:
        return ""
    desc, emoji = _wx_desc(daily.get("code"))
    tmax = daily.get("tmax")
    tmin = daily.get("tmin")
    temp = ""
    if tmax is not None and tmin is not None:
        temp = f'{tmin:.0f}~{tmax:.0f}℃'
    elif tmax is not None:
        temp = f'{tmax:.0f}℃'
    sr = _hhmm(daily.get("sunrise"))
    ss = _hhmm(daily.get("sunset"))
    psum = daily.get("precip_sum")
    pprob = daily.get("precip_prob")
    rain = ""
    if pprob is not None and psum is not None:
        rain = f'降水 {pprob:.0f}% · {psum:.1f}mm'
    elif pprob is not None:
        rain = f'降水概率 {pprob:.0f}%'
    parts = [f'{emoji} {desc}']
    if temp:
        parts.append(f'🌡️ {temp}')
    if sr:
        parts.append(f'🌅 {sr}')
    if ss:
        parts.append(f'🌇 {ss}')
    if rain:
        parts.append(f'💧 {rain}')
    return (
        f'<div style="padding:7px 16px;font-size:12px;color:#475569;'
        f'background:#fffbeb;border-bottom:1px solid #fef3c7;'
        f'display:flex;flex-wrap:wrap;gap:6px 14px;line-height:1.6;">'
        + "".join(f'<span>{html.escape(p)}</span>' for p in parts)
        + '</div>'
    )


def _day_card(r):
    w = r["windows"]
    date_str = dt.datetime.strptime(r["date"], "%Y-%m-%d")
    weekday = "周" + "一二三四五六日"[date_str.weekday()]
    title = f'{r["date"]} {weekday}'

    # 找出当天最好时段，用于高亮
    best_window = max(w.items(), key=lambda kv: kv[1]["final"])[0]

    morning = w.get("morning")
    evening = w.get("evening")
    blocks = ""
    if morning is not None:
        blocks += _window_block("朝霞", "🌅", morning)
    if evening is not None:
        blocks += _window_block("晚霞", "🌇", evening)

    weather_line = _daily_weather_line(r.get("daily"))

    return f'''
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05);">
      <div style="padding:14px 16px;background:linear-gradient(90deg,#fdf2f8,#fff7ed);font-weight:800;font-size:16px;color:#1e293b;display:flex;justify-content:space-between;">
        <span>{title}</span>
        <span style="font-size:12px;font-weight:600;color:#f97316;">最佳:{'朝霞' if best_window == 'morning' else '晚霞'}</span>
      </div>
      {weather_line}
      {blocks}
    </div>'''


_SHARE_CARD_HTML = r"""
  <!-- 分享卡片弹层 -->
  <div id="shareModal" style="display:none;position:fixed;inset:0;background:rgba(15,23,42,.72);z-index:999;align-items:center;justify-content:center;padding:16px;">
    <div style="background:#fff;border-radius:16px;padding:18px;width:100%;max-width:640px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.3);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="font-weight:700;color:#1e293b;font-size:15px;">分享卡片</span>
        <button onclick="closeShare()" style="border:none;background:#f1f5f9;border-radius:8px;padding:5px 12px;cursor:pointer;color:#475569;font-size:13px;">关闭</button>
      </div>
      <canvas id="shareCanvas" width="900" height="600" style="width:100%;border-radius:12px;display:block;background:#f1f5f9;"></canvas>
      <div style="margin-top:12px;">
        <a id="shareDownload" download="朝霞晚霞预测.png" href="#" style="display:inline-block;background:#f97316;color:#fff;padding:9px 24px;border-radius:999px;text-decoration:none;font-size:14px;font-weight:600;">下载图片</a>
        <div style="font-size:11px;color:#94a3b8;margin-top:8px;">手机端长按图片可直接保存分享</div>
      </div>
    </div>
  </div>

  <script>
    const BEST_CARD = __BEST_CARD_JSON__;

    function openShare() {
      if (!BEST_CARD) return;
      drawShareCard(BEST_CARD);
      document.getElementById('shareModal').style.display = 'flex';
    }
    function closeShare() {
      document.getElementById('shareModal').style.display = 'none';
    }

    function scoreColor(s) {
      if (s >= 75) return '#f97316';
      if (s >= 60) return '#fbbf24';
      if (s >= 45) return '#94a3b8';
      return '#cbd5e1';
    }

    function rr(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function drawShareCard(c) {
      const canvas = document.getElementById('shareCanvas');
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const font = "'PingFang SC','Microsoft YaHei',sans-serif";

      // 背景渐变（火烧云色）
      const g = ctx.createLinearGradient(0, 0, W, H);
      g.addColorStop(0, '#fb7185');
      g.addColorStop(0.5, '#f97316');
      g.addColorStop(1, '#8b5cf6');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);

      // 装饰光晕圆
      ctx.globalAlpha = 0.18;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath(); ctx.arc(780, 90, 150, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(90, 540, 110, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;

      // 白色圆角面板
      ctx.shadowColor = 'rgba(0,0,0,.18)';
      ctx.shadowBlur = 30;
      ctx.shadowOffsetY = 10;
      ctx.fillStyle = '#ffffff';
      rr(ctx, 40, 40, W - 80, H - 80, 28);
      ctx.fill();
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      const px = 90, pw = W - 180; // 面板内文字起始 x 与宽度
      let y = 110;

      // 小标签
      ctx.fillStyle = '#94a3b8';
      ctx.font = '22px ' + font;
      ctx.fillText('朝霞 · 晚霞预测', px, y);
      y += 52;

      // 城市名
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 54px ' + font;
      ctx.fillText(c.city, px, y);
      y += 62;

      // 日期 + 时段
      ctx.fillStyle = '#475569';
      ctx.font = '26px ' + font;
      ctx.fillText(c.date + ' · ' + c.when + '（' + c.window + '）', px, y);
      y += 40;

      // 分隔线
      ctx.strokeStyle = '#f1f5f9';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, y); ctx.lineTo(W - px, y); ctx.stroke();
      y += 46;

      // 大分数
      const sc = scoreColor(c.score);
      ctx.fillStyle = sc;
      ctx.font = 'bold 96px ' + font;
      ctx.fillText(String(c.score), px, y + 10);
      const scoreW = ctx.measureText(String(c.score)).width;
      ctx.fillStyle = '#94a3b8';
      ctx.font = '26px ' + font;
      ctx.fillText('/100', px + scoreW + 12, y + 10);
      // 分级 + 鲜艳度
      ctx.fillStyle = '#c2410c';
      ctx.font = 'bold 30px ' + font;
      ctx.fillText(c.vivid + ' · ' + c.grade, px + scoreW + 90, y + 10);
      ctx.fillStyle = '#7c3aed';
      ctx.font = 'bold 26px ' + font;
      ctx.fillText('🔥 ' + c.chroma + '/10', px, y + 52);
      y += 78;

      // 一句话提示（自动换行）
      ctx.fillStyle = '#334155';
      ctx.font = '24px ' + font;
      wrapText(ctx, c.note, px, y, pw, 34, 2);
      y += 34 * (c.note.length > 20 ? 2 : 1) + 10;

      // 因素行
      const aod = (c.aod !== null && c.aod !== undefined) ? ('AOD ' + c.aod) : 'AOD 暂无';
      const vis = (c.vis !== null && c.vis !== undefined) ? ('能见度 ' + c.vis + 'km') : '能见度 暂无';
      ctx.fillStyle = '#64748b';
      ctx.font = '20px ' + font;
      ctx.fillText('云量 ' + c.cloud + '% · 湿度 ' + c.humidity + '% · 风 ' + c.wind + 'm/s', px, y);
      y += 30;
      ctx.fillText(aod + ' · ' + vis, px, y);

      // 底部署名
      ctx.fillStyle = '#94a3b8';
      ctx.font = '16px ' + font;
      ctx.textAlign = 'right';
      ctx.fillText('数据源 Open-Meteo · 仅供参考', W - px, H - 80);
      ctx.textAlign = 'left';

      document.getElementById('shareDownload').href = canvas.toDataURL('image/png');
    }

    function wrapText(ctx, text, x, y, maxWidth, lineHeight, maxLines) {
      const chars = (text || '').split('');
      let line = '';
      const lines = [];
      for (let i = 0; i < chars.length; i++) {
        const test = line + chars[i];
        if (ctx.measureText(test).width > maxWidth && line) {
          lines.push(line);
          line = chars[i];
          if (lines.length === maxLines) break;
        } else {
          line = test;
        }
      }
      if (line && lines.length < maxLines) lines.push(line);
      lines.forEach(function (l, idx) { ctx.fillText(l, x, y + idx * lineHeight); });
    }
  </script>
"""


def render_report(cfg, results, meta, train_result=None):
    """渲染完整 HTML 报告，返回 HTML 字符串。"""
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    city = meta["city"]

    # 找全局最佳
    best = None
    for r in results:
        for w, f in r["windows"].items():
            if best is None or f["final"] > best[1]["final"]:
                best = ((r["date"], w), f)

    # 组装「分享卡片」所需数据（前端 Canvas 绘制用）
    best_card = None
    if best:
        (bd, bw), bf = best
        aod = bf.get("aod")
        vis = bf.get("visibility")
        best_card = {
            "city": city,
            "date": bd,
            "window": "朝霞" if bw == "morning" else "晚霞",
            "when": "清晨" if bw == "morning" else "傍晚",
            "score": round(bf["final"], 0),
            "grade": bf["grade"],
            "vivid": vividness_of(bf["final"]),
            "chroma": bf.get("chroma") if bf.get("chroma") is not None else chroma_index(bf["final"]),
            "note": factor_note(bf["breakdown"]),
            "cloud": round(bf["cloud_cover"], 0),
            "humidity": round(bf["humidity"], 0),
            "wind": round(bf["wind"], 1),
            "aod": round(aod, 2) if aod is not None else None,
            "vis": round(vis / 1000, 1) if vis is not None else None,
        }
    best_card_json = json.dumps(best_card, ensure_ascii=False)

    cards = "\n".join(_day_card(r) for r in results)

    # 模型状态
    if train_result and train_result.get("trained"):
        s = train_result["stats"]
        m = train_result["metrics"]
        real = s.get("posts", 0)
        model_note = (f'模型已训练 · 样本 {s["total"]}（真实观测 {real} 条'
                      f' + 弱监督 {s["total"] - real} 条）')
        # 只展示留出集指标 —— 训练集指标是"能否复刻规则"，不具预测意义
        if "holdout_auc" in m or "holdout_accuracy" in m:
            bits = []
            if "holdout_accuracy" in m:
                bits.append(f'留出集准确率 {m["holdout_accuracy"]}')
            if "holdout_auc" in m:
                bits.append(f'留出集 AUC {m["holdout_auc"]}')
            model_note += (" · " + " · ".join(bits)
                           + f'（时序留出 {m.get("n_test", 0)} 条真实观测）')
        elif m.get("holdout_note"):
            model_note += f' · {m["holdout_note"]}'
    elif meta.get("model_gated"):
        model_note = (
            f'真实观测标注 {meta.get("model_real_labels", 0)} 条'
            f' < 门槛 {meta.get("min_real_labels", 30)} 条，已按设计退回纯规则评分 '
            f'—— 弱监督标签由规则分本身生成，用它算出的指标只能说明'
            f'"能否复刻规则引擎"，不能说明真实预测能力'
        )
    elif meta.get("has_model"):
        model_note = "使用已保存模型预测"
    else:
        model_note = (
            "当前为纯规则评分模式。模型启用的条件是真实观测标注达到门槛"
            "（config.yaml 的 model.min_real_labels，默认 30 条）；"
            "在 data/raw/posts.csv 补充真实观测后会自动启用"
        )

    best_line = ""
    if best:
        (d, w), f = best
        best_line = (
            f'<div style="margin:16px 0 8px;font-size:15px;color:#1e293b;">'
            f'<strong style="color:#f97316;">✨ 最佳机会</strong>：'
            f'{d} {("清晨" if w == "morning" else "傍晚")}，'
            f'评分 <strong>{f["final"]:.0f}</strong>（{f["grade"]}）'
            f'</div>'
        )

    html_str = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{city}朝霞晚霞预测</title>
</head>
<body style="margin:0;background:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;">
  <div style="max-width:720px;margin:0 auto;padding:24px 16px 48px;">
    <div style="background:linear-gradient(135deg,#fb7185,#f97316,#8b5cf6);border-radius:20px;padding:28px 24px;color:#fff;">
      <div style="font-size:13px;opacity:.9;">朝霞 · 晚霞 概率预测</div>
      <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
        <div>
          <div style="font-size:28px;font-weight:800;margin-top:4px;">{html.escape(city)}</div>
          <div style="font-size:13px;opacity:.9;margin-top:6px;">更新于 {now} · 数据源 Open-Meteo</div>
        </div>
        <button onclick="openShare()" style="flex-shrink:0;border:1px solid rgba(255,255,255,.7);background:rgba(255,255,255,.15);color:#fff;padding:9px 16px;border-radius:999px;font-size:13px;font-weight:600;cursor:pointer;transition:all .15s;" onmouseover="this.style.background='rgba(255,255,255,.3)'" onmouseout="this.style.background='rgba(255,255,255,.15)'">📤 生成分享卡片</button>
      </div>
    </div>

    {best_line}

    <div style="display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px;">
      {cards}
    </div>

    <div style="margin-top:20px;padding:14px 16px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;font-size:12px;color:#64748b;line-height:1.7;">
      <div><strong style="color:#334155;">说明</strong></div>
      <div>· 评分 0-100，≥75 优秀、60-75 良好、45-60 一般、&lt;45 平淡。</div>
      <div>· 鲜艳度分级（对齐 SunsetBot）：微烧/小烧/中烧/大烧/优质大烧/世纪大烧；🔥 为 0-10 鲜艳度指数（对齐 chromasky）。</div>
      <div>· 评分由 10 因子加权：云结构/低云遮挡/湿度/降水/风速/气溶胶AOD/能见度（基础因子合计 84%），加上方向性的「剖面云边界」「太阳方位遮挡」「地形遮蔽」（合计 16%）。</div>
      <div>· <strong style="color:#334155;">缺数据 = 不表态</strong>：任何一个因子取不到数（AOD 接口失败、模式缺分层云量等），该因子会被<strong>整体移出加权</strong>，其余因子按比例重新归一化 —— 而不是给一个"中间分"，因为带权重的中间分依然会牵动总分。</div>
      <div>· 气溶胶(AOD)：清洁天气下 AOD 越低、能见度越高，天空越通透、火烧云越鲜艳；但若<strong>沙尘主导</strong>（dust/PM10 高），AOD 与出霞的关系转为非单调 —— 约 0.25 附近反而最出彩，因为沙尘的前向散射像一层暖色滤镜。</div>
      <div>· <strong style="color:#334155;">雨洗效应</strong>：窗口前 24h 若有大雨（≥10mm），气溶胶被湿沉降冲刷约 50%，通透度实际优于预报值；若是微量降水，反而因颗粒吸湿增长使通透度略降。</div>
      <div>· <strong style="color:#334155;">地形遮蔽</strong>：太阳方位上的地平线仰角（PVGIS 90m 数字高程）。西边有山时可见的低空天空被切掉一块，而那些天空正是低角度火烧云所在。</div>
      <div>· <strong style="color:#334155;">方向性</strong>：朝霞晚霞的光来自太阳所在方位，因此程序会沿日出/日落方位角拉一条 0~500km 剖面。「剖面云边界」衡量云层边缘的位置 —— 云幕在约 400km 处到达边缘时，边缘以外的晴空让低角度阳光从云层下方斜射进来，把近处云底整体点亮，这是最壮观火烧云的成因；「太阳方位遮挡」则看太阳方向近场(0~150km)的低云会不会把整条光路切断。</div>
      <div>· 日落瞬间太阳在地平线下约 0.83°，受地球曲率阴影限制，1km 的低云此时已变暗（阴影角约 1.02°），而 5km 以上的中高云仍在接光（约 2.27°）—— 这是"火烧云多是中高云"的几何原因。同一几何还给出：此刻阳光与地面相切于约 <strong>185km</strong>，比这更近处的云底完全照不到光。</div>
      <div>· {html.escape(model_note)}。</div>
      <div>· 交叉验证：用 GFS(美)/ICON(德)/GEM(加) 三个独立气象模式互相印证，并按其均值做多源集成打分（占 30% 权重）；分差越小、置信度越高，分差大说明该时段云况不稳定。</div>
      <div>· 朝霞/晚霞受局地云况影响大，预报仅供参考，出门前请结合实际天空状况判断。</div>
    </div>
  </div>
__SHARE_CARD__
</body>
</html>'''
    return (html_str
            .replace("__SHARE_CARD__", _SHARE_CARD_HTML)
            .replace("__BEST_CARD_JSON__", best_card_json))


def save_report(cfg, results, meta, train_result=None):
    """渲染并写入报告文件，返回文件路径。"""
    import os

    html_str = render_report(cfg, results, meta, train_result)
    report_path = cfg["output"]["report"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return report_path
