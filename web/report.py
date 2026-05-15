"""Bundle a session's outputs into a single .zip for one-click download."""
from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Optional


def build_zip(session_dir: str, metrics: dict, summary_text: str,
              screenshot_path: Optional[str] = None) -> bytes:
    """Returns the .zip bytes containing seg.nii.gz + metrics.json +
    summary.txt + (optional) screenshot.png."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seg_path = os.path.join(session_dir, "seg.nii.gz")
        if os.path.exists(seg_path):
            zf.write(seg_path, arcname="seg.nii.gz")
        zf.writestr("metrics.json", json.dumps(metrics, indent=2))
        zf.writestr("summary.txt", summary_text)
        if screenshot_path and os.path.exists(screenshot_path):
            zf.write(screenshot_path, arcname="screenshot.png")
    buf.seek(0)
    return buf.read()
