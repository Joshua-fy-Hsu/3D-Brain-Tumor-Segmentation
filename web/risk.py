"""Risk level from population-derived percentiles of BraTS ground-truth volumes.

`web/data/population_stats.json` is precomputed by
scripts/prepare_webapp_assets.py over the full cohort (all patients with a
GT mask — 1251 by default; pass --population-scope val for the held-out
split only). Schema:

    {
      "ET": {"p10":..,"p33":..,"p67":..,"p90":..,"n":1251,"values":[..]},
      "TC": {...},
      "WT": {...}
    }

Level thresholds: `v <= p33` → Low, `<= p67` → Medium, `<= p90` → High,
else Very High. Percentile via empirical CDF using "values" if present.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(HERE, "data", "population_stats.json")

LEVEL_COLORS = {
    "Low": "#2d8f2d",
    "Medium": "#d9b026",
    "High": "#d97a26",
    "Very High": "#c0392b",
    "Unknown": "#888888",
}


class Population:
    def __init__(self, stats: Optional[dict]):
        self.stats = stats  # may be None if file is missing
        if stats is not None:
            self._sorted = {
                region: sorted(d.get("values", [])) for region, d in stats.items()
            }
        else:
            self._sorted = {}

    @property
    def available(self) -> bool:
        return self.stats is not None


def load_population() -> Population:
    if not os.path.exists(STATS_PATH):
        log.warning(f"population_stats.json not found at {STATS_PATH} — "
                    "risk level disabled. Run scripts/prepare_webapp_assets.py.")
        return Population(None)
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        return Population(json.load(f))


def _percentile(sorted_vals: list, v: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = bisect.bisect_right(sorted_vals, v)
    return round(100.0 * idx / len(sorted_vals), 1)


def classify_one(region: str, v_ml: float, pop: Population) -> dict:
    if not pop.available or region not in pop.stats:
        return {
            "level": "Unknown",
            "color": LEVEL_COLORS["Unknown"],
            "percentile": None,
            "volume_ml": round(float(v_ml), 2),
        }
    d = pop.stats[region]
    if v_ml <= d["p33"]:
        level = "Low"
    elif v_ml <= d["p67"]:
        level = "Medium"
    elif v_ml <= d["p90"]:
        level = "High"
    else:
        level = "Very High"
    return {
        "level": level,
        "color": LEVEL_COLORS[level],
        "percentile": _percentile(pop._sorted.get(region, []), v_ml),
        "volume_ml": round(float(v_ml), 2),
    }


def classify(volumes: dict[str, float], pop: Population) -> dict:
    """volumes = {"ET": .., "TC": .., "WT": ..} in mL. Returns per-region dicts."""
    return {region: classify_one(region, float(v), pop) for region, v in volumes.items()}
