"""Server FastAPI untuk Pakcoy Vision."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import (
    DEFAULT_PIXELS_PER_CM,
    MAX_UPLOAD_MB,
    SAMPLES,
    SAMPLES_DIR,
    STATIC_DIR,
    YOLO_CONF,
    YOLO_IOU,
    ensure_directories,
)
from .pipeline import Pipeline

ensure_directories()

app = FastAPI(
    title="Pakcoy Vision",
    description=(
        "Pipeline YOLOv11-seg + OpenCV + klasifikasi fase pertumbuhan pakcoy "
        "hidroponik."
    ),
    version=__version__,
)

pipeline = Pipeline()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _sample_payload() -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for sample in SAMPLES:
        image_path = SAMPLES_DIR / str(sample["file"])
        thumb_path = SAMPLES_DIR / f"{sample['id']}-thumb.jpg"
        item: dict[str, object] = dict(sample)
        item["url"] = f"/static/samples/{image_path.name}"
        item["thumb"] = (
            f"/static/samples/{thumb_path.name}" if thumb_path.exists() else None
        )
        payload.append(item)
    return payload


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/health", include_in_schema=False)
def health_alias() -> dict[str, str]:
    """Alias untuk health check layanan/monitor eksternal."""
    return health()


@app.get("/api/status")
def status() -> dict[str, object]:
    return {
        "app": {
            "name": "Pakcoy Vision",
            "version": __version__,
        },
        **pipeline.status(),
        "samples": _sample_payload(),
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "default_conf": YOLO_CONF,
            "default_iou": YOLO_IOU,
            "default_pixels_per_cm": DEFAULT_PIXELS_PER_CM,
        },
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    conf: float = Form(YOLO_CONF),
    iou: float = Form(YOLO_IOU),
    pixels_per_cm: str = Form(""),
    engine: str = Form("auto"),
) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Berkas harus berupa citra (JPG, PNG, atau WEBP).",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Berkas citra kosong.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran berkas melebihi {MAX_UPLOAD_MB} MB.",
        )

    # Kalibrasi hanya aktif jika pengguna mengisi nilainya. Default tidak
    # memakai angka apa pun supaya pengukuran tampil dalam piksel dan tidak
    # terlihat seolah sudah tervalidasi.
    calibration: float | None = None
    if str(pixels_per_cm).strip():
        try:
            calibration = float(pixels_per_cm.replace(",", "."))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Kalibrasi piksel/cm harus berupa angka.",
            ) from exc
        if calibration <= 0:
            raise HTTPException(
                status_code=422,
                detail="Kalibrasi piksel/cm harus lebih besar dari 0.",
            )
    if calibration is not None and calibration <= 0:
        calibration = None

    if engine not in {"auto", "yolo", "classic"}:
        raise HTTPException(
            status_code=422,
            detail="Pilihan mesin segmentasi tidak dikenal.",
        )
    if not 0.01 <= conf <= 0.99:
        raise HTTPException(
            status_code=422,
            detail="Confidence threshold harus berada di antara 0,01 dan 0,99.",
        )
    if not 0.05 <= iou <= 0.95:
        raise HTTPException(
            status_code=422,
            detail="IoU threshold harus berada di antara 0,05 dan 0,95.",
        )

    try:
        result = await run_in_threadpool(
            pipeline.analyze,
            data,
            file.filename or "foto-pakcoy.jpg",
            conf,
            iou,
            calibration,
            engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - jalur error runtime
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline analisis gagal: {exc}",
        ) from exc

    return result


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
