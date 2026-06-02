"""Cumulative, cross-session usage ledger for the sustainability dashboard.

A single JSON file under `web/_stats/cumulative.json` accumulates the running
totals shown on the workstation's "Sustainability" panel (inferences served,
energy, CO2e, electricity cost, and radiologist-minutes saved). It survives
restarts and the 24 h session sweep — unlike `web/_sessions/`, this ledger is
never swept.

Writes are serialised with a module lock and committed atomically (tmp +
os.replace) so a crash mid-write can't corrupt the file.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(HERE, "_stats")
STATS_PATH = os.path.join(STATS_DIR, "cumulative.json")

_lock = threading.Lock()

_DEFAULT = {
    "total_inferences": 0,
    "total_energy_wh": 0.0,
    "total_co2_g": 0.0,
    "total_cost_twd": 0.0,
    "total_manual_minutes_saved": 0.0,
    "since": None,
    "updated": None,
}


def _read_locked() -> dict:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in _DEFAULT.items():
            d.setdefault(k, v)
        return d
    except Exception:
        return dict(_DEFAULT)


def _write_locked(d: dict) -> None:
    os.makedirs(STATS_DIR, exist_ok=True)
    tmp = STATS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STATS_PATH)


def record(energy_wh: float, co2_g: float, cost_twd: float,
           manual_minutes: float) -> dict:
    """Add one inference's footprint to the ledger; return the new snapshot."""
    with _lock:
        d = _read_locked()
        today = time.strftime("%Y-%m-%d")
        if not d.get("since"):
            d["since"] = today
        d["total_inferences"] = int(d["total_inferences"]) + 1
        d["total_energy_wh"] = float(d["total_energy_wh"]) + float(energy_wh)
        d["total_co2_g"] = float(d["total_co2_g"]) + float(co2_g)
        d["total_cost_twd"] = float(d["total_cost_twd"]) + float(cost_twd)
        d["total_manual_minutes_saved"] = (
            float(d["total_manual_minutes_saved"]) + float(manual_minutes))
        d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_locked(d)
        return d


def snapshot() -> dict:
    with _lock:
        return _read_locked()


def reset() -> dict:
    """Zero the ledger (one-click reset for demos). Returns the cleared state."""
    with _lock:
        d = dict(_DEFAULT)
        _write_locked(d)
        return d
