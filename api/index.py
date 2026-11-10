"""Entrypoint serverless Vercel untuk Pakcoy Vision.

Deployment ini hanya menjalankan mode segmentasi klasik (HSV + ExG + Otsu +
watershed). YOLOv11-seg tidak disertakan karena torch + ultralytics melebihi
batas 250 MB per fungsi Vercel. Untuk mode model terlatih, gunakan Dockerfile
(Hugging Face Spaces / VPS).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402,F401
