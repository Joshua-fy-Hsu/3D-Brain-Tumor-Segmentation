"""中文版自動摘要產生器。

純樣板字串（不使用 LLM）。消化流程前段產出的結構化欄位，輸出一段
可直接貼入病歷的純文字段落。對應原版 `web/summary.py`。
"""
from __future__ import annotations

from typing import Optional

LOW_CONF_THRESHOLD = 0.70

# risk.classify 回傳的英文等級 → 中文
LEVEL_ZH = {
    "Low": "低",
    "Medium": "中",
    "High": "高",
    "Very High": "極高",
    "Unknown": "未知",
}


def _fmt_vol(v: Optional[float]) -> str:
    return "0.0 mL" if v is None else f"{v:.1f} mL"


def _fmt_conf(c: Optional[float]) -> str:
    return "無" if c is None else f"{c:.2f}"


def build(volumes: dict, anatomy: list[dict], confidence: dict, risk: dict,
          malignancy: Optional[dict] = None) -> str:
    """回傳摘要段落文字。

    volumes:   {"ET": float, "TC": float, "WT": float}，單位 mL。
    anatomy:   依重疊比例由大到小排序的 {"name", "pct"} 清單（前 K 名）。
    confidence:{"ET": Optional[float], "TC":..., "WT":...}，範圍 [0,1]。
    risk:      risk.classify 的輸出（每區 {"level","percentile",...}）。
    malignancy:malignancy.assess 的輸出（影像特徵指標）或 None。
    """
    wt_v = volumes.get("WT", 0.0)
    tc_v = volumes.get("TC", 0.0)
    et_v = volumes.get("ET", 0.0)

    wt_risk = risk.get("WT", {})
    level_zh = LEVEL_ZH.get(wt_risk.get("level", "Unknown"), wt_risk.get("level", "未知"))
    pct_str = ""
    if wt_risk.get("percentile") is not None:
        pct_str = f"（第 {wt_risk['percentile']:.0f} 百分位，風險{level_zh}）"
    elif wt_risk.get("level") and wt_risk["level"] != "Unknown":
        pct_str = f"（風險{level_zh}）"

    parts = []
    parts.append(
        f"預測全腫瘤體積 {_fmt_vol(wt_v)}{pct_str}，"
        f"腫瘤核心 {_fmt_vol(tc_v)}，強化腫瘤 {_fmt_vol(et_v)}。"
    )

    if anatomy:
        top = anatomy[:3]
        loc = "、".join(f"{a['name']}（{a['pct']:.0f}%）" for a in top)
        parts.append(f"腫瘤主要侵犯 {loc}。")
    else:
        parts.append("此影像無法提供解剖定位。")

    parts.append(
        f"各區平均信心：全腫瘤 {_fmt_conf(confidence.get('WT'))}、"
        f"腫瘤核心 {_fmt_conf(confidence.get('TC'))}、"
        f"強化腫瘤 {_fmt_conf(confidence.get('ET'))}。"
    )

    low = [r for r in ("WT", "TC", "ET")
           if confidence.get(r) is not None and confidence[r] < LOW_CONF_THRESHOLD]
    if low:
        name_map = {"WT": "全腫瘤", "TC": "腫瘤核心", "ET": "強化腫瘤"}
        low_names = "／".join(name_map[r] for r in low)
        parts.append(
            f"警告：{low_names} 信心低於 {LOW_CONF_THRESHOLD:.2f} 門檻，"
            f"判讀時請謹慎。"
        )

    if malignancy and malignancy.get("category") not in (None, "unknown"):
        label_zh = malignancy.get("label_zh", "")
        parts.append(
            f"影像特徵提示：{label_zh}。此為依分割結果推導之描述性影像指標，"
            f"並非腫瘤分級／分期，且未經病理驗證，不可作為診斷依據。"
        )

    return " ".join(parts)
