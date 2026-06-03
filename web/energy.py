"""Per-inference energy / carbon / cost accounting for the workstation.

Sustainability instrumentation for the demo. Every `/api/predict` call is
wrapped in an `EnergyMeter`, which samples real GPU power draw during the
sliding-window forward pass and integrates it to watt-hours. From energy we
derive CO2e (Taiwan grid emission factor) and electricity cost (Taipower
tariff), plus a hospital-scale extrapolation that frames the much larger win:
radiologist hours saved versus manual tumour contouring.

Power-telemetry backends, tried in order:
  1. pynvml  — in-process NVML, lowest overhead (used if installed).
  2. nvidia-smi --query-gpu=power.draw -lms 100 — streamed subprocess sampler
     (works out-of-the-box on the dev RTX 4060 Laptop; pynvml is not installed).
  3. estimate — fixed TDP fallback when no GPU telemetry is available at all
     (clearly flagged measured=False so the UI never over-claims precision).

All numbers are intentionally *gross* GPU energy over the inference window
(idle draw is not subtracted), which slightly over-estimates our footprint —
the honest direction. Conversion factors are env-overridable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Optional

# ── Conversion constants (env-overridable) ──────────────────────────────────
# Taiwan grid emission factor — Bureau of Energy 2023 電力排碳係數: 0.494 kgCO2e/kWh.
GRID_CO2_KG_PER_KWH = float(os.environ.get("GRID_CO2_KG_PER_KWH", "0.494"))
# Taipower average tariff (NT$/kWh) — ~3.5 is a representative blended rate.
ELECTRICITY_TWD_PER_KWH = float(os.environ.get("ELECTRICITY_TWD_PER_KWH", "3.5"))
# Minutes for full manual 3D voxel-wise delineation of all tumour sub-regions
# (NCR/ED/ET over the whole volume) — the research-grade task this model
# actually replaces. Literature puts this at ~1–4 h/case; we default to the
# conservative low value of 1 h. Override via MANUAL_SEG_MINUTES.
MANUAL_SEG_MINUTES = float(os.environ.get("MANUAL_SEG_MINUTES", "60"))
# Fallback GPU power if no telemetry is available (RTX 4060 Laptop ~ 75 W busy).
FALLBACK_GPU_TDP_W = float(os.environ.get("FALLBACK_GPU_TDP_W", "75"))

# Scale-up / equivalence reference points.
HOSPITAL_CASES_PER_DAY = float(os.environ.get("HOSPITAL_CASES_PER_DAY", "50"))
TREE_KG_CO2_PER_YEAR = 21.0       # mature tree CO2 sequestration, kg/yr
CAR_KG_CO2_PER_KM = 0.192         # avg passenger car, 192 gCO2/km

_SAMPLE_INTERVAL_S = 0.1


# ── Backend discovery (done once at import) ─────────────────────────────────
def _init_pynvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        return pynvml, handle, name
    except Exception:
        return None, None, None


_PYNVML, _NVML_HANDLE, _NVML_NAME = _init_pynvml()
_SMI = shutil.which("nvidia-smi")


def _smi_gpu_name() -> Optional[str]:
    if not _SMI:
        return None
    try:
        out = subprocess.run(
            [_SMI, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = out.stdout.strip().splitlines()[0].strip()
        return name or None
    except Exception:
        return None


if _PYNVML is not None:
    BACKEND = "nvml"
    GPU_NAME = _NVML_NAME or "GPU"
elif _SMI:
    BACKEND = "nvidia-smi"
    GPU_NAME = _smi_gpu_name() or "GPU"
else:
    BACKEND = None
    GPU_NAME = "CPU/estimate"


class EnergyMeter:
    """Context manager that records GPU energy over the wrapped block.

    Usage:
        meter = EnergyMeter()
        with meter:
            run_inference()
        meter.reading  # dict, ready after __exit__

    Never raises into the wrapped block on telemetry failure — it silently
    degrades to the TDP estimate so a prediction is never blocked by metering.
    """

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self.t0 = 0.0
        self.duration_s = 0.0
        self.reading: dict = {}

    # -- backend sampling loops ------------------------------------------
    def _nvml_loop(self) -> None:
        while not self._stop.is_set():
            try:
                w = _PYNVML.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
                with self._lock:
                    self._samples.append(w)
            except Exception:
                pass
            self._stop.wait(_SAMPLE_INTERVAL_S)

    def _smi_loop(self) -> None:
        try:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    w = float(line)
                except ValueError:
                    continue
                with self._lock:
                    self._samples.append(w)
        except Exception:
            pass

    # -- context protocol ------------------------------------------------
    def __enter__(self) -> "EnergyMeter":
        self.t0 = time.perf_counter()
        try:
            if BACKEND == "nvml":
                self._reader = threading.Thread(target=self._nvml_loop, daemon=True)
                self._reader.start()
            elif BACKEND == "nvidia-smi":
                self._proc = subprocess.Popen(
                    [_SMI, "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits",
                     "-lms", str(int(_SAMPLE_INTERVAL_S * 1000))],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                )
                self._reader = threading.Thread(target=self._smi_loop, daemon=True)
                self._reader.start()
        except Exception:
            self._reader = None
            self._proc = None
        return self

    def __exit__(self, *exc) -> bool:
        self.duration_s = max(time.perf_counter() - self.t0, 1e-6)
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self.reading = self._compute()
        return False  # never suppress exceptions

    # -- result ----------------------------------------------------------
    def _compute(self) -> dict:
        with self._lock:
            samples = list(self._samples)
        if samples:
            mean_w = sum(samples) / len(samples)
            measured = True
            method = BACKEND
        else:
            mean_w = FALLBACK_GPU_TDP_W
            measured = False
            method = "estimate"

        energy_wh = mean_w * self.duration_s / 3600.0
        kwh = energy_wh / 1000.0
        co2_g = kwh * GRID_CO2_KG_PER_KWH * 1000.0
        cost_twd = kwh * ELECTRICITY_TWD_PER_KWH

        return {
            "measured": measured,
            "method": method,
            "backend_name": GPU_NAME,
            "duration_s": round(self.duration_s, 2),
            "mean_power_w": round(mean_w, 1),
            "samples": len(samples),
            "energy_wh": round(energy_wh, 4),
            "co2_g": round(co2_g, 3),
            "cost_twd": round(cost_twd, 5),
            "manual_minutes_saved": MANUAL_SEG_MINUTES,
            "scale": scale_up(energy_wh, co2_g, cost_twd),
        }


def scale_up(energy_wh: float, co2_g: float, cost_twd: float) -> dict:
    """Extrapolate one inference to a hospital running it at clinical volume."""
    cases = HOSPITAL_CASES_PER_DAY * 365.0
    e_kwh = energy_wh / 1000.0 * cases
    co2_kg = co2_g / 1000.0 * cases
    cost = cost_twd * cases
    return {
        "cases_per_day": HOSPITAL_CASES_PER_DAY,
        "cases_per_year": round(cases),
        "energy_kwh": round(e_kwh, 2),
        "co2_kg": round(co2_kg, 2),
        "cost_twd": round(cost, 1),
        "manual_hours_saved": round(cases * MANUAL_SEG_MINUTES / 60.0),
        "equiv_car_km": round(co2_kg / CAR_KG_CO2_PER_KM, 1),
        "equiv_tree_years": round(co2_kg / TREE_KG_CO2_PER_YEAR, 2),
    }


def usage_summary(snap: dict) -> dict:
    """Enrich a cumulative-usage snapshot with derived totals + equivalences."""
    e_kwh = snap.get("total_energy_wh", 0.0) / 1000.0
    co2_kg = snap.get("total_co2_g", 0.0) / 1000.0
    return {
        **snap,
        "total_energy_kwh": round(e_kwh, 4),
        "total_co2_kg": round(co2_kg, 4),
        "total_manual_hours_saved": round(snap.get("total_manual_minutes_saved", 0.0) / 60.0, 1),
        "equiv_car_km": round(co2_kg / CAR_KG_CO2_PER_KM, 2),
        "equiv_tree_years": round(co2_kg / TREE_KG_CO2_PER_YEAR, 3),
        "grid_co2_kg_per_kwh": GRID_CO2_KG_PER_KWH,
        "price_twd_per_kwh": ELECTRICITY_TWD_PER_KWH,
        "telemetry": BACKEND or "estimate",
        "gpu_name": GPU_NAME,
    }
