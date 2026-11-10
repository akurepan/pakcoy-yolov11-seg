"""Konfigurasi terpusat untuk aplikasi Pakcoy Vision."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
SAMPLES_DIR = STATIC_DIR / "samples"
ASSETS_DIR = STATIC_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# Ukuran sisi terpanjang citra. Samakan dengan preprocessing saat training
# agar distribusi fitur numerik tetap konsisten.
INFER_MAX_SIZE = 640
MAX_UPLOAD_MB = 20
JPEG_QUALITY = 88

YOLO_CONF = 0.25
YOLO_IOU = 0.50
MIN_LEAF_AREA_RATIO = 0.0006  # terhadap luas citra
MAX_LEAF_AREA_RATIO = 0.72
MAX_LEAVES = 40

# Kalibrasi tetap agar hasil morfologi langsung tampil dalam cm.
# Nilai bawaan 38,5 px/cm hanya untuk uji tampilan dan BELUM tervalidasi.
# Ganti dengan hasil kalibrasi penggaris, atau set environment variable
# PAKCOY_PIXELS_PER_CM (mis. 41.2) tanpa mengubah kode.
DEFAULT_PIXELS_PER_CM = 38.5
_env_pixels_per_cm = os.environ.get("PAKCOY_PIXELS_PER_CM", "").strip()
if _env_pixels_per_cm:
    try:
        DEFAULT_PIXELS_PER_CM = float(_env_pixels_per_cm.replace(",", "."))
    except ValueError:
        pass

PHASES = ["Penyemaian", "Vegetatif", "Siap Panen"]
PHASE_TO_AGE = {
    "Penyemaian": "0–14 HST",
    "Vegetatif": "15–30 HST",
    "Siap Panen": "31–45 HST",
}
PHASE_INDEX = {name: idx for idx, name in enumerate(PHASES)}

FEATURE_COLUMNS = [
    "leaf_count",
    "mean_length",
    "mean_width",
    "max_length",
    "max_width",
    "total_area",
    "mean_ratio",
]

# Kandidat bobot YOLOv11-seg hasil training. Bisa juga diarahkan lewat
# environment variable PAKCOY_YOLO_WEIGHTS.
# `models/best.pt` adalah model terbaru yang dipakai aplikasi, jadi selalu
# dicoba lebih dulu. Kandidat lain hanya cadangan.
YOLO_WEIGHT_CANDIDATES = [
    MODELS_DIR / "best.pt",
    MODELS_DIR / "pakcoy_leaf_seg.pt",
    MODELS_DIR / "yolo11s-seg-pakcoy.pt",
    MODELS_DIR / "pakcoy_yolo11s_seg.pt",
    BASE_DIR / "hasil" / "yolo11n-seg.pt",
    # Checkpoint hasil training yang disimpan bersama dataset.
    BASE_DIR / "dataset" / "dataset-pakcoy.pt",
]

# Ukuran inferensi harus sama dengan saat training (imgsz 640).
YOLO_IMGSZ = 640

TFLITE_CANDIDATES = [MODELS_DIR / "pakcoy_mlp.tflite"]
SCALER_CANDIDATES = [MODELS_DIR / "feature_scaler.pkl"]
LABEL_ENCODER_CANDIDATES = [MODELS_DIR / "label_encoder.pkl"]
SKLEARN_CANDIDATES = [
    MODELS_DIR / "best_mlp_sklearn.pkl",
    MODELS_DIR / "best_rf.pkl",
]

MODEL_INFO_FILE = MODELS_DIR / "model_info.json"

SAMPLES = [
    {
        "id": "day-05",
        "file": "day-05.jpg",
        "label": "Hari 5",
        "hint": "Penyemaian",
        "day": 5,
    },
    {
        "id": "day-20",
        "file": "day-20.jpg",
        "label": "Hari 20",
        "hint": "Vegetatif",
        "day": 20,
    },
    {
        "id": "day-30",
        "file": "day-30.jpg",
        "label": "Hari 30",
        "hint": "Vegetatif akhir",
        "day": 30,
    },
    {
        "id": "day-42",
        "file": "day-42.jpg",
        "label": "Hari 42",
        "hint": "Siap panen",
        "day": 42,
    },
]

# Palet instance (BGR) agar tiap daun punya warna berbeda.
PALETTE_BGR = [
    (102, 209, 255),
    (160, 214, 6),
    (178, 138, 17),
    (111, 71, 239),
    (107, 140, 247),
    (250, 139, 167),
    (222, 241, 244),
    (238, 211, 34),
    (133, 113, 251),
    (22, 204, 132),
    (181, 122, 132),
    (73, 190, 176),
    (120, 216, 154),
    (199, 179, 255),
    (87, 108, 255),
    (112, 166, 46),
]

PALETTE_HEX = [
    "#%02x%02x%02x" % (r, g, b) for (b, g, r) in PALETTE_BGR
]


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def find_yolo_weights() -> Path | None:
    override = os.environ.get("PAKCOY_YOLO_WEIGHTS")
    if override:
        path = Path(override)
        if path.exists():
            return path
    return first_existing(YOLO_WEIGHT_CANDIDATES)


def ensure_directories() -> None:
    """Buat folder yang dibutuhkan bila belum ada.

    Filesystem serverless (mis. Vercel) bersifat read-only, jadi kegagalan
    membuat folder diabaikan selama folder tersebut sudah ada di bundel.
    """
    for directory in (MODELS_DIR, SAMPLES_DIR, ASSETS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
