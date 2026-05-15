"""FastAPI web GUI for brain tumor segmentation inference.

A single-page workstation that accepts a 4-modality MRI upload, runs the
pinned `full` variant via sliding-window inference, and returns volumes,
anatomy involvement, confidence, risk level, a summary paragraph, and the
NIfTI files for in-browser Niivue rendering.

Launch with:
    python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
"""
