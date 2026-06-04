"""Auto-written summary paragraph builder.

Pure templating — no LLM. Consumes the structured fields produced earlier
in the pipeline and emits a plain-text paragraph suitable for copy-pasting
into a medical record.
"""
from __future__ import annotations

from typing import Optional

LOW_CONF_THRESHOLD = 0.70


def _fmt_vol(v: Optional[float]) -> str:
    return "0.0 mL" if v is None else f"{v:.1f} mL"


def _fmt_conf(c: Optional[float]) -> str:
    return "n/a" if c is None else f"{c:.2f}"


def build(volumes: dict, anatomy: list[dict], confidence: dict, risk: dict,
          malignancy: Optional[dict] = None) -> str:
    """Returns the paragraph text.

    volumes:   {"ET": float, "TC": float, "WT": float} in mL.
    anatomy:   list of {"name", "pct"} sorted by pct desc, top-K.
    confidence:{"ET": Optional[float], "TC":..., "WT":...} in [0,1].
    risk:      output of risk.classify (per-region {"level","percentile",...}).
    malignancy:output of malignancy.assess (imaging-feature indicator) or None.
    """
    wt_v = volumes.get("WT", 0.0)
    tc_v = volumes.get("TC", 0.0)
    et_v = volumes.get("ET", 0.0)

    wt_risk = risk.get("WT", {})
    pct_str = ""
    if wt_risk.get("percentile") is not None:
        pct_str = f" ({wt_risk['percentile']:.0f}th percentile, {wt_risk.get('level', 'Unknown')})"
    elif wt_risk.get("level") and wt_risk["level"] != "Unknown":
        pct_str = f" ({wt_risk['level']})"

    parts = []
    parts.append(
        f"Predicted whole tumor volume {_fmt_vol(wt_v)}{pct_str}, "
        f"tumor core {_fmt_vol(tc_v)}, enhancing tumor {_fmt_vol(et_v)}."
    )

    if anatomy:
        top = anatomy[:3]
        loc = ", ".join(f"{a['name']} ({a['pct']:.0f}%)" for a in top)
        parts.append(f"Tumor primarily involves {loc}.")
    else:
        parts.append("Anatomical localisation unavailable for this volume.")

    parts.append(
        f"Mean per-region confidence: WT {_fmt_conf(confidence.get('WT'))}, "
        f"TC {_fmt_conf(confidence.get('TC'))}, ET {_fmt_conf(confidence.get('ET'))}."
    )

    low = [r for r in ("WT", "TC", "ET")
           if confidence.get(r) is not None and confidence[r] < LOW_CONF_THRESHOLD]
    if low:
        parts.append(
            f"Warning: {'/'.join(low)} confidence below {LOW_CONF_THRESHOLD:.2f} "
            f"threshold — interpret with caution."
        )

    if malignancy and malignancy.get("category") not in (None, "unknown"):
        drivers = malignancy.get("drivers") or []
        drv = f" ({'; '.join(drivers)})" if drivers else ""
        parts.append(
            f"Imaging-feature note: {malignancy.get('label', '')}{drv}. "
            f"This is a descriptive imaging indicator only, not a tumour "
            f"grade/stage and not validated against pathology."
        )

    return " ".join(parts)
