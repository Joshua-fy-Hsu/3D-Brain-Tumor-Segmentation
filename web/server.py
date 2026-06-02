"""FastAPI app: serves the static UI and the /api endpoints.

Run with:
    python -m uvicorn web.server:app --host 0.0.0.0 --port 8000

On startup we:
  - sweep stale session directories,
  - load the pinned `full` variant + its newest compatible checkpoint,
  - load the AAL atlas (optional) and the population stats (optional).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Annotated

import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from web import anatomy as A
from web import energy as EN
from web import inference as I
from web import metrics_case as MC
from web import preprocess as P
from web import report as RP
from web import risk as RK
from web import state as S
from web import summary as SUM
from web import usage as USG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("web")

app = FastAPI(title="Brain Tumor Segmentation Workstation")


@app.middleware("http")
async def _no_cache(request: Request, call_next):
    """Disable caching for the UI so edits to static files always show
    without a hard refresh (this is a single-user demo app)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static") or path == "/api/meta":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


STATIC_DIR = os.path.join(HERE, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Module-level singletons populated at startup.
_loaded_model: I.LoadedModel | None = None
_atlas: A.Atlas | None = None
_population: RK.Population | None = None


@app.on_event("startup")
def _on_startup() -> None:
    global _loaded_model, _atlas, _population
    os.makedirs(S.SESSIONS_ROOT, exist_ok=True)
    removed = S.sweep_old_sessions()
    if removed:
        log.info(f"swept {removed} stale session(s)")

    log.info(f"loading model variant='{I.VARIANT_NAME}' ...")
    _loaded_model = I.load_model()
    log.info(f"loaded checkpoint: {_loaded_model.checkpoint_path} "
             f"(arch_family→{_loaded_model.amp_dtype}, mode={_loaded_model.output_mode})")

    _atlas = A.load_atlas()
    log.info(f"atlas available={_atlas.available} labels={len(_atlas.labels)}")

    _population = RK.load_population()
    log.info(f"population stats available={_population.available}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/meta")
def meta() -> dict:
    """UI calls this on load to show the current model run name."""
    if _loaded_model is None:
        return {"ready": False}
    return {
        "ready": True,
        "model_name": I.MODEL_DISPLAY_NAME,
        "model_version": I.MODEL_VERSION,
        "device": "GPU" if "cuda" in str(_loaded_model.device) else "CPU",
        "atlas": _atlas.available if _atlas else False,
        "population": _population.available if _population else False,
    }


@app.get("/api/stats")
def stats() -> dict:
    """Cumulative sustainability ledger for the dashboard panel."""
    return EN.usage_summary(USG.snapshot())


def _save_nifti(arr: np.ndarray, affine: np.ndarray, path: str) -> None:
    nib.save(nib.Nifti1Image(arr, affine), path)


@app.post("/api/predict")
async def predict(
    t1: Annotated[UploadFile, File(...)],
    t1ce: Annotated[UploadFile, File(...)],
    t2: Annotated[UploadFile, File(...)],
    flair: Annotated[UploadFile, File(...)],
) -> JSONResponse:
    if _loaded_model is None:
        raise HTTPException(503, "Model not ready")

    sid, sdir = S.new_session()
    uploads = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    saved_paths: dict[str, str] = {}
    for name, up in uploads.items():
        fname = up.filename or f"{name}.nii.gz"
        ext = ".nii.gz" if fname.lower().endswith(".nii.gz") else (
              ".nii" if fname.lower().endswith(".nii") else ".nii.gz")
        out_path = os.path.join(sdir, f"{name}{ext}")
        with open(out_path, "wb") as f:
            f.write(await up.read())
        saved_paths[name] = out_path

    try:
        case = P.build_input(saved_paths, device=_loaded_model.device)
    except P.PreprocessError as e:
        raise HTTPException(400, str(e))

    log.info(f"[{sid}] preprocessed shape={case.spatial_shape} "
             f"voxel={case.voxel_volume_ml:.4f} mL")

    meter = EN.EnergyMeter()
    try:
        with meter:
            res = I.run(_loaded_model, case.x)
    except Exception as e:
        log.exception(f"[{sid}] inference failed")
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        if case.x.is_cuda:
            torch.cuda.empty_cache()

    energy = meter.reading
    USG.record(energy["energy_wh"], energy["co2_g"], energy["cost_twd"],
               energy["manual_minutes_saved"])
    log.info(f"[{sid}] energy={energy['energy_wh']} Wh "
             f"co2={energy['co2_g']} g via {energy['method']} "
             f"({energy['samples']} samples)")

    seg_canonical_path = os.path.join(sdir, "seg.nii.gz")
    _save_nifti(res.labels.astype(np.uint8), case.ref_affine, seg_canonical_path)
    canonical_modality_paths = {}
    for m in P.MODALITIES:
        dst = os.path.join(sdir, f"{m}.nii.gz")
        if dst != saved_paths[m]:
            try:
                os.replace(saved_paths[m], dst)
            except OSError:
                import shutil
                shutil.copyfile(saved_paths[m], dst)
        canonical_modality_paths[m] = dst

    volumes = MC.region_volumes_ml(res.labels, case.voxel_volume_ml)
    confidence = MC.region_confidence(res.probs_4ch, res.labels)
    anatomy_top = A.overlap_top_k(res.labels, _atlas, k=5) if _atlas else []
    risk = RK.classify(volumes, _population) if _population else {}
    summary_text = SUM.build(volumes, anatomy_top, confidence, risk)

    metrics = {
        "session_id": sid,
        "variant": _loaded_model.variant,
        "run_name": _loaded_model.run_name,
        "spatial_shape": list(case.spatial_shape),
        "voxel_volume_ml": case.voxel_volume_ml,
        "volumes_ml": {k: round(float(v), 3) for k, v in volumes.items()},
        "confidence": confidence,
        "risk": risk,
        "anatomy_top": anatomy_top,
        "summary": summary_text,
        "energy": energy,
    }
    with open(os.path.join(sdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(sdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # Kick off MC-Dropout uncertainty in a background thread so the seg
    # result is returned immediately. The frontend polls /uncertainty_status.
    x_cpu = case.x.cpu()  # move off GPU before handing to thread
    affine_copy = case.ref_affine.copy()

    def _compute_uncertainty():
        try:
            x_dev = x_cpu.to(_loaded_model.device)
            ent = I.run_uncertainty(_loaded_model, x_dev, T=10)
            if ent is not None:
                # Save as uncompressed .nii — avoids uvicorn re-gzipping an
                # already-gzipped file (Content-Length mismatch / RuntimeError).
                _save_nifti(ent, affine_copy,
                            os.path.join(sdir, "uncertainty.nii"))
                log.info(f"[{sid}] uncertainty map saved")
        except Exception:
            log.exception(f"[{sid}] uncertainty computation failed")
        finally:
            if x_dev.is_cuda:
                del x_dev
                torch.cuda.empty_cache()

    threading.Thread(target=_compute_uncertainty, daemon=True).start()

    return JSONResponse({
        **metrics,
        "nifti_urls": {
            "t1": f"/api/session/{sid}/t1.nii.gz",
            "t1ce": f"/api/session/{sid}/t1ce.nii.gz",
            "t2": f"/api/session/{sid}/t2.nii.gz",
            "flair": f"/api/session/{sid}/flair.nii.gz",
            "seg": f"/api/session/{sid}/seg.nii.gz",
        },
        "uncertainty_url": f"/api/session/{sid}/uncertainty_status",
    })


# NOTE: the specific routes (/screenshot, /report) MUST be declared before
# the catch-all /{fname} route — FastAPI matches in declaration order, so a
# generic /{fname} placed first would swallow /report as fname="report".
@app.post("/api/session/{sid}/screenshot")
async def upload_screenshot(sid: str, request: Request) -> dict:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "Session not found")
    blob = await request.body()
    if not blob:
        raise HTTPException(400, "Empty body")
    if len(blob) > 20 * 1024 * 1024:
        raise HTTPException(413, "Screenshot too large")
    with open(os.path.join(sdir, "screenshot.png"), "wb") as f:
        f.write(blob)
    return {"ok": True}


@app.get("/api/session/{sid}/report")
def download_report(sid: str) -> Response:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "Session not found")
    metrics_path = os.path.join(sdir, "metrics.json")
    summary_path = os.path.join(sdir, "summary.txt")
    if not (os.path.exists(metrics_path) and os.path.exists(summary_path)):
        raise HTTPException(404, "Report files missing — run /api/predict first")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
    screenshot_path = os.path.join(sdir, "screenshot.png")
    blob = RP.build_zip(sdir, metrics, summary_text,
                       screenshot_path if os.path.exists(screenshot_path) else None)
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="report_{sid[:8]}.zip"'},
    )


@app.get("/api/session/{sid}/uncertainty_status")
def uncertainty_status(sid: str) -> JSONResponse:
    """Returns whether the MC-Dropout uncertainty map is ready."""
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "Session not found")
    path = os.path.join(sdir, "uncertainty.nii")
    ready = os.path.exists(path)
    return JSONResponse({
        "ready": ready,
        "url": f"/api/session/{sid}/uncertainty.nii" if ready else None,
    })


# Catch-all session file route — declared LAST so it does not shadow the
# specific /screenshot and /report routes above.
@app.get("/api/session/{sid}/{fname}")
def session_file(sid: str, fname: str) -> FileResponse:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "Session not found")
    if "/" in fname or "\\" in fname or fname.startswith("."):
        raise HTTPException(400, "Bad filename")
    path = os.path.join(sdir, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path)
