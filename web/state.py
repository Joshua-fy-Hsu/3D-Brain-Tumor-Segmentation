"""Session directory lifecycle for the web app.

Each `/api/predict` call gets a uuid4 directory under `web/_sessions/`.
The dir holds the user's uploaded NIfTI files, the predicted segmentation,
a metrics.json, summary.txt, and optionally a screenshot.png posted by the
client before the report-zip download.

On startup we sweep entries older than `SESSION_TTL_HOURS`.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_ROOT = os.path.join(HERE, "_sessions")
SESSION_TTL_HOURS = 24


def new_session() -> tuple[str, str]:
    """Returns (session_id, absolute_path). Creates the directory."""
    sid = uuid.uuid4().hex
    path = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(path, exist_ok=False)
    return sid, path


def session_path(sid: str) -> Optional[str]:
    """Return the dir for `sid` if it exists and is under SESSIONS_ROOT."""
    if not _is_valid_sid(sid):
        return None
    path = os.path.join(SESSIONS_ROOT, sid)
    if not os.path.isdir(path):
        return None
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(SESSIONS_ROOT) + os.sep):
        return None
    return path


def _is_valid_sid(sid: str) -> bool:
    return bool(sid) and len(sid) == 32 and all(c in "0123456789abcdef" for c in sid)


def sweep_old_sessions() -> int:
    """Delete sessions older than SESSION_TTL_HOURS. Returns count removed."""
    if not os.path.isdir(SESSIONS_ROOT):
        os.makedirs(SESSIONS_ROOT, exist_ok=True)
        return 0
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    removed = 0
    for name in os.listdir(SESSIONS_ROOT):
        path = os.path.join(SESSIONS_ROOT, name)
        if not os.path.isdir(path):
            continue
        try:
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
