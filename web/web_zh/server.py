"""FastAPI app（中文版）：提供靜態介面與 /api 端點。

啟動方式：
    python -m uvicorn web.web_zh.server:app --host 0.0.0.0 --port 8001

這是原版 `web/server.py` 的中文版本，差異：
  - 提供 web_zh/static 的中文前端，
  - 摘要文字改用 web_zh.summary_zh，
  - 移除預測不確定性 (MC-Dropout uncertainty map) 的背景計算與相關端點。

推論、前處理、解剖、風險、報告等邏輯全部重用原版 `web` 套件，原版完全不受影響。
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
ROOT = os.path.dirname(os.path.dirname(HERE))   # web/web_zh -> repo root
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

# 重用原版 web 套件的推論模組（不修改原版）。
from web import anatomy as A
from web import energy as EN
from web import inference as I
from web import malignancy as MAL
from web import metrics_case as MC
from web import preprocess as P
from web import report_pdf as RPDF
from web import risk as RK
from web import state as S
from web import usage as USG
from web.web_zh import summary_zh as SUM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("web_zh")

app = FastAPI(title="腦腫瘤分割工作站")


@app.middleware("http")
async def _no_cache(request: Request, call_next):
    """關閉 UI 快取，讓靜態檔案的修改不需硬重新整理即可生效（單人示範用）。"""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static") or path == "/api/meta":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


STATIC_DIR = os.path.join(HERE, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 啟動時填入的模組級單例。
_loaded_model: I.LoadedModel | None = None
_atlas: A.Atlas | None = None
_population: RK.Population | None = None


@app.on_event("startup")
def _on_startup() -> None:
    global _loaded_model, _atlas, _population
    os.makedirs(S.SESSIONS_ROOT, exist_ok=True)
    removed = S.sweep_old_sessions()
    if removed:
        log.info(f"清除了 {removed} 個過期 session")

    log.info(f"載入模型 variant='{I.VARIANT_NAME}' ...")
    _loaded_model = I.load_model()
    log.info(f"已載入 checkpoint: {_loaded_model.checkpoint_path} "
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
    """UI 載入時呼叫，用來顯示目前模型名稱。"""
    if _loaded_model is None:
        return {"ready": False}
    return {
        "ready": True,
        "model_name": "TumorSeg",
        "model_version": I.MODEL_VERSION,
        "device": "GPU" if "cuda" in str(_loaded_model.device) else "CPU",
        "atlas": _atlas.available if _atlas else False,
        "population": _population.available if _population else False,
    }


@app.get("/api/stats")
def stats() -> dict:
    """永續儀表板的累積用量帳本。"""
    return EN.usage_summary(USG.snapshot())


@app.post("/api/stats/reset")
def stats_reset() -> dict:
    """一鍵將累積帳本歸零（demo 用）。"""
    log.info("累積用量帳本已透過 /api/stats/reset 歸零")
    return EN.usage_summary(USG.reset())


def _save_nifti(arr: np.ndarray, affine: np.ndarray, path: str) -> None:
    nib.save(nib.Nifti1Image(arr, affine), path)


def _patient_id(orig_names: dict[str, str]) -> str:
    """從上傳檔名推導病患代號，例如
    'BraTS2021_00495_t1ce.nii.gz' -> 'BraTS2021_00495'；
    若為通用檔名（t1ce.nii.gz）則回傳空字串。"""
    import re
    for key in ("t1ce", "flair", "t2", "t1"):
        fn = orig_names.get(key) or ""
        base = fn
        for ext in (".nii.gz", ".nii"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
                break
        base = re.sub(r"[_\- ]?(t1ce|t1c|t1|t2|flair|seg)$", "", base, flags=re.I)
        base = base.strip(" _-")
        if base and base.lower() not in ("t1ce", "t1c", "t1", "t2", "flair"):
            return base
    return ""


@app.post("/api/predict")
async def predict(
    t1: Annotated[UploadFile, File(...)],
    t1ce: Annotated[UploadFile, File(...)],
    t2: Annotated[UploadFile, File(...)],
    flair: Annotated[UploadFile, File(...)],
) -> JSONResponse:
    if _loaded_model is None:
        raise HTTPException(503, "模型尚未就緒")

    sid, sdir = S.new_session()
    uploads = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    orig_names = {name: (up.filename or "") for name, up in uploads.items()}
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
        raise HTTPException(500, f"推論失敗: {e}")
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
    # 由前景通道（5 通道輸入的第 4 索引）計算腦容積。
    brain_ml = float((case.x[0, 4] > 0).sum().item()) * case.voxel_volume_ml
    volume_pct = MC.region_volume_pct(volumes, brain_ml)
    confidence = MC.region_confidence(res.probs_4ch, res.labels)
    anatomy_top = A.overlap_top_k(res.labels, _atlas, k=5) if _atlas else []
    risk = RK.classify(volumes, _population) if _population else {}
    malignancy = MAL.assess(res.labels, case.voxel_volume_ml)
    summary_text = SUM.build(volumes, anatomy_top, confidence, risk, malignancy)

    metrics = {
        "session_id": sid,
        "patient_id": _patient_id(orig_names),
        "variant": _loaded_model.variant,
        "run_name": _loaded_model.run_name,
        "spatial_shape": list(case.spatial_shape),
        "voxel_volume_ml": case.voxel_volume_ml,
        "volumes_ml": {k: round(float(v), 3) for k, v in volumes.items()},
        "volume_pct": volume_pct,
        "brain_volume_ml": round(brain_ml, 1),
        "confidence": confidence,
        "risk": risk,
        "malignancy": malignancy,
        "anatomy_top": anatomy_top,
        "summary": summary_text,
        "energy": energy,
    }
    with open(os.path.join(sdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(sdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    # 在背景執行緒計算 MC-Dropout 預測不確定性，立即回傳分割結果。
    # 前端會輪詢 /uncertainty_status。
    x_cpu = case.x.cpu()
    affine_copy = case.ref_affine.copy()

    def _compute_uncertainty():
        x_dev = None
        try:
            x_dev = x_cpu.to(_loaded_model.device)
            ent = I.run_uncertainty(_loaded_model, x_dev, T=10)
            if ent is not None:
                # 存為未壓縮 .nii — 避免 uvicorn 重新 gzip 已壓縮檔。
                _save_nifti(ent, affine_copy,
                            os.path.join(sdir, "uncertainty.nii"))
                log.info(f"[{sid}] uncertainty map saved")
        except Exception:
            log.exception(f"[{sid}] uncertainty computation failed")
        finally:
            if x_dev is not None and x_dev.is_cuda:
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


# 注意：特定路由 (/screenshot, /report) 必須宣告在 catch-all /{fname} 之前 ——
# FastAPI 依宣告順序比對，泛用的 /{fname} 若在前面會吃掉 /report。
@app.post("/api/session/{sid}/screenshot")
async def upload_screenshot(sid: str, request: Request) -> dict:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    blob = await request.body()
    if not blob:
        raise HTTPException(400, "內容為空")
    if len(blob) > 20 * 1024 * 1024:
        raise HTTPException(413, "截圖過大")
    with open(os.path.join(sdir, "screenshot.png"), "wb") as f:
        f.write(blob)
    return {"ok": True}


@app.post("/api/session/{sid}/slices")
async def upload_slices(sid: str, request: Request) -> dict:
    """切面檢視器（橫切／冠狀／矢狀 MRI 切片）截圖 —— 一併打包進報告 zip。"""
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    blob = await request.body()
    if not blob:
        raise HTTPException(400, "內容為空")
    if len(blob) > 20 * 1024 * 1024:
        raise HTTPException(413, "截圖過大")
    with open(os.path.join(sdir, "slices.png"), "wb") as f:
        f.write(blob)
    return {"ok": True}


@app.post("/api/session/{sid}/uncertainty_shot")
async def upload_uncertainty_shot(sid: str, request: Request) -> dict:
    """預測不確定性檢視器截圖 —— 一併打包進報告 PDF。"""
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    blob = await request.body()
    if not blob:
        raise HTTPException(400, "內容為空")
    if len(blob) > 20 * 1024 * 1024:
        raise HTTPException(413, "截圖過大")
    with open(os.path.join(sdir, "uncertainty.png"), "wb") as f:
        f.write(blob)
    return {"ok": True}


@app.get("/api/session/{sid}/report")
def download_report(sid: str) -> Response:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    metrics_path = os.path.join(sdir, "metrics.json")
    summary_path = os.path.join(sdir, "summary.txt")
    if not (os.path.exists(metrics_path) and os.path.exists(summary_path)):
        raise HTTPException(404, "報告檔案不存在 —— 請先執行 /api/predict")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_text = f.read()
    metrics["model_name"] = "TumorSeg"
    screenshot_path = os.path.join(sdir, "screenshot.png")
    slices_path = os.path.join(sdir, "slices.png")
    uncertainty_path = os.path.join(sdir, "uncertainty.png")
    blob = RPDF.build_pdf(
        metrics, summary_text, lang="zh",
        screenshot_path=screenshot_path if os.path.exists(screenshot_path) else None,
        slices_path=slices_path if os.path.exists(slices_path) else None,
        uncertainty_path=uncertainty_path if os.path.exists(uncertainty_path) else None,
    )
    return Response(
        content=blob,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{sid[:8]}.pdf"'},
    )


@app.get("/api/session/{sid}/uncertainty_status")
def uncertainty_status(sid: str) -> JSONResponse:
    """回傳 MC-Dropout 不確定性圖是否計算完成。"""
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    path = os.path.join(sdir, "uncertainty.nii")
    ready = os.path.exists(path)
    return JSONResponse({
        "ready": ready,
        "url": f"/api/session/{sid}/uncertainty.nii" if ready else None,
    })


# Catch-all session 檔案路由 —— 宣告在最後，避免遮蔽上面的特定路由。
@app.get("/api/session/{sid}/{fname}")
def session_file(sid: str, fname: str) -> FileResponse:
    sdir = S.session_path(sid)
    if sdir is None:
        raise HTTPException(404, "找不到 session")
    if "/" in fname or "\\" in fname or fname.startswith("."):
        raise HTTPException(400, "檔名不合法")
    path = os.path.join(sdir, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "找不到檔案")
    return FileResponse(path)
