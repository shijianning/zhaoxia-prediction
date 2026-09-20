"""微信推送模块（Server酱）。

融合 ohyep-sunsetglow / sunsetbot 的推送能力：当预测出现达到阈值的
"火烧云"窗口时，通过 Server酱 推送到微信，方便出门前/下班前及时知晓。

未在 config.yaml 配置 notify.sckey 或 enabled=false 时静默跳过，不影响预测。
"""
import requests

from .glow_rules import vividness_of

PUSH_URL = "https://sctapi.ftqq.com/{sckey}.send"


def _find_best(results):
    """从预测结果里找全局最高分窗口。返回 (date, window, feature) 或 None。"""
    best = None
    for r in results:
        for w, f in r["windows"].items():
            if best is None or f["final"] > best[2]["final"]:
                best = (r["date"], w, f)
    return best


def _threshold_for(window, n):
    """取该时段阈值：优先 morning/evening 独立阈值，否则回退到统一 threshold。"""
    key = "morning_threshold" if window == "morning" else "evening_threshold"
    if key in n and n[key] is not None:
        return float(n[key])
    return float(n.get("threshold", 70))


def _confidence_adjust(f):
    """根据多模型一致性微调阈值：结论一致更可信 → 阈值可略降；分歧大 → 阈值上浮。

    返回相对基线的增量（正=更严格，负=更宽松）。
    """
    xc = f.get("xcheck") or {}
    conf = xc.get("confidence")
    if conf == "高":
        return -5
    if conf == "低":
        return 5
    return 0


def send_notification(cfg, results, meta):
    """把达到阈值的最佳窗口推送到微信。

    阈值策略：
    - 支持朝霞/晚霞独立阈值（morning_threshold / evening_threshold）；
    - 结合多模型一致性（spread→confidence）动态微调 ±5 分。
    返回状态描述字符串；未启用/未配置时返回 None（调用方据此决定是否打印）。
    """
    n = cfg.get("notify") or {}
    if not n.get("enabled") or not n.get("sckey"):
        return None

    best = _find_best(results)
    if best is None:
        return "无可用预测窗口，跳过推送"

    date, window, f = best
    base_thr = _threshold_for(window, n)
    thr = base_thr + _confidence_adjust(f)
    if f["final"] < thr:
        return (f"最高分 {f['final']:.0f} 未达推送阈值 {thr:.0f}"
                f"（{('朝霞' if window == 'morning' else '晚霞')}阈值 {base_thr:.0f}），不推送")

    tag = "朝霞" if window == "morning" else "晚霞"
    title = f"{meta['city']}{tag}预警：{f['grade']} {f['final']:.0f}分"
    lines = [
        f"日期：{date}",
        f"时段：{tag}",
        f"评分：{f['final']:.0f}/100（{f['grade']} · {vividness_of(f['final'])}）",
    ]
    chroma = f.get("chroma")
    if chroma is not None:
        lines.append(f"鲜艳度指数：{chroma}/10")
    xc = f.get("xcheck") or {}
    if xc.get("confidence"):
        lines.append(f"多模型一致性：{xc.get('confidence')}（分差 {xc.get('spread')}）")
    desp = "\n".join(lines)

    try:
        resp = requests.post(
            PUSH_URL.format(sckey=n["sckey"]),
            data={"title": title, "desp": desp},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            return "微信推送成功"
        return f"微信推送返回异常：{data}"
    except Exception as exc:
        return f"微信推送失败：{exc}"


def send_failure_alert(cfg, error):
    """脚本运行失败时推送一次告警（供 logging/入口调用）。未启用时返回 None。"""
    n = cfg.get("notify") or {}
    if not n.get("enabled") or not n.get("sckey"):
        return None
    city = (cfg.get("city") or {}).get("name", "")
    try:
        resp = requests.post(
            PUSH_URL.format(sckey=n["sckey"]),
            data={
                "title": f"{city}朝霞晚霞预测运行失败",
                "desp": f"脚本运行出错：\n{error}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return "失败告警已推送" if data.get("code") == 0 else f"告警推送异常：{data}"
    except Exception as exc:
        return f"告警推送失败：{exc}"
