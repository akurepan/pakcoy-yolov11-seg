"""Pipeline inti: preprocessing, segmentasi, ekstraksi fitur, dan agregasi."""

from __future__ import annotations

import base64
import importlib.util
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classifier import PhaseClassifier
from .config import (
    FEATURE_COLUMNS,
    INFER_MAX_SIZE,
    JPEG_QUALITY,
    MAX_LEAF_AREA_RATIO,
    MAX_LEAVES,
    MIN_LEAF_AREA_RATIO,
    PALETTE_BGR,
    PALETTE_HEX,
    PHASE_TO_AGE,
    YOLO_IMGSZ,
    find_yolo_weights,
)
from .recommendations import build_recommendations


@dataclass
class Instance:
    mask: np.ndarray
    confidence: float | None = None
    source: str = "classic"


@dataclass
class Leaf:
    index: int
    mask: np.ndarray
    contour: np.ndarray
    rect: tuple
    length_px: float
    width_px: float
    area_px: float
    ratio: float
    confidence: float | None
    centroid: tuple[float, float]
    color_bgr: tuple[int, int, int]
    color_hex: str
    width_visible_px: float = 0.0
    occluded: bool = False


@dataclass
class SegmentResult:
    instances: list[Instance] = field(default_factory=list)
    engine: str = "classic"
    label: str = "Segmentasi Klasik (demo)"
    note: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilitas citra
# ---------------------------------------------------------------------------


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Berkas tidak dapat dibaca sebagai citra. Gunakan JPG, PNG, atau WEBP."
        )
    return image


def resize_max(image: np.ndarray, max_size: int = INFER_MAX_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize → CLAHE → Gaussian blur ringan (sesuai diagram alur)."""
    resized = resize_max(image)
    enhanced = apply_clahe(resized)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return resized, denoised


def encode_data_url(image: np.ndarray, mime: str = "image/jpeg") -> str:
    if mime == "image/png":
        ok, buffer = cv2.imencode(".png", image)
    else:
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
    if not ok:
        raise RuntimeError("Gagal mengodekan citra hasil.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Segmentasi klasik (fallback demo)
# ---------------------------------------------------------------------------


def classic_segment(image: np.ndarray) -> list[Instance]:
    """Segmentasi daun berbasis warna + watershed.

    Dipakai sebagai mode demo ketika bobot YOLOv11-seg hasil training belum
    tersedia. Tujuannya agar seluruh pipeline tetap dapat diuji end-to-end.
    """
    height, width = image.shape[:2]
    image_area = float(height * width)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (25, 35, 25), (98, 255, 255))

    b_channel, g_channel, r_channel = cv2.split(image.astype(np.int16))
    excess_green = 2 * g_channel - r_channel - b_channel
    exg_norm = cv2.normalize(excess_green, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    _, exg_mask = cv2.threshold(
        exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_and(green, exg_mask)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Buang komponen hijau yang terlalu kecil (noise) atau terlalu besar
    # (mis. dinding/background hijau).
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < image_area * 0.0003:
            continue
        if area > image_area * 0.92:
            continue
        cleaned[labels == label] = 255

    if cv2.countNonZero(cleaned) == 0:
        return []

    distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return []
    distance_norm = distance / distance.max()
    sure_foreground = np.uint8(distance_norm > 0.28) * 255

    seed_count, seeds = cv2.connectedComponents(sure_foreground)
    if seed_count <= 1:
        # Tidak ada puncak jarak yang valid: pakai kontur apa adanya.
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        instances: list[Instance] = []
        for contour in contours:
            instance_mask = np.zeros_like(cleaned)
            cv2.drawContours(instance_mask, [contour], -1, 255, -1)
            instances.append(Instance(mask=instance_mask > 0, source="classic"))
        return instances

    markers = seeds.astype(np.int32) + 1
    unknown = cv2.subtract(cleaned, sure_foreground)
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    instances = []
    min_area = max(160.0, image_area * MIN_LEAF_AREA_RATIO)
    max_area = image_area * MAX_LEAF_AREA_RATIO

    for label in range(2, int(markers.max()) + 1):
        instance_mask = (markers == label) & (cleaned > 0)
        area = float(instance_mask.sum())
        if area < min_area or area > max_area:
            continue
        contours, _ = cv2.findContours(
            instance_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area * 0.75:
            continue
        instances.append(Instance(mask=instance_mask, source="classic"))

    return instances


# ---------------------------------------------------------------------------
# YOLOv11-seg
# ---------------------------------------------------------------------------


def masks_from_polygons(
    polygons: list[np.ndarray],
    height: int,
    width: int,
    confidences: list[float | None] | None = None,
) -> list[Instance]:
    """Rasterisasi poligon mask YOLO ke resolusi citra asli.

    `masks.xy` dari Ultralytics sudah berada di koordinat citra asli. Memakai
    poligon ini menghindari pergeseran yang muncul ketika tensor mask letterbox
    di-resize langsung ke ukuran citra.
    """
    instances: list[Instance] = []
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        if cv2.countNonZero(mask) < 30:
            continue
        confidence = None
        if confidences and index < len(confidences):
            confidence = confidences[index]
        instances.append(
            Instance(
                mask=mask > 0,
                confidence=float(confidence) if confidence is not None else None,
                source="yolo",
            )
        )
    return instances


class YoloSegmenter:
    """Pembungkus YOLOv11-seg dengan metadata checkpoint dan hot reload.

    Bobot dibaca ulang otomatis ketika berkas `best.pt` diganti, sehingga model
    terbaru langsung dipakai tanpa mematikan server. Mask instance dirasterisasi
    dari poligon `masks.xy` pada resolusi citra asli, bukan dari tensor
    letterbox, agar hasil tidak bergeser pada foto non-persegi.
    """

    def __init__(self) -> None:
        self.weights_path = find_yolo_weights()
        self.runtime_available = (
            importlib.util.find_spec("ultralytics") is not None
        )
        self._model = None
        self._model_signature: tuple[str, int, int] | None = None
        self._meta: dict[str, Any] | None = None
        self._meta_signature: tuple[str, int, int] | None = None
        self._load_error: str | None = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.weights_path is not None and self.runtime_available

    # ------------------------------------------------------------------
    # Metadata checkpoint
    # ------------------------------------------------------------------

    @staticmethod
    def _signature(path: Path | None) -> tuple[str, int, int] | None:
        if path is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def _refresh_path(self) -> None:
        current = find_yolo_weights()
        if current != self.weights_path:
            self.weights_path = current
            self._model = None
            self._model_signature = None
            self._load_error = None

    def model_info(self) -> dict[str, Any] | None:
        """Ringkasan checkpoint: arsitektur, kelas, dan metrik validasi."""
        self._refresh_path()
        path = self.weights_path
        signature = self._signature(path)
        if path is None or signature is None:
            return None
        with self._lock:
            if signature != self._meta_signature:
                self._meta = self._read_checkpoint_metadata(path)
                self._meta_signature = signature
            return self._meta

    @staticmethod
    def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
        info: dict[str, Any] = {
            "file": path.name,
            "architecture": None,
            "task": "segment",
            "classes": [],
            "imgsz": YOLO_IMGSZ,
            "epochs": None,
            "trained_at": None,
            "metrics": None,
        }
        try:
            import torch

            checkpoint = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
        except Exception:
            return info

        model = checkpoint.get("model")
        train_args = checkpoint.get("train_args") or {}
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            info["classes"] = [str(name) for name in names.values()]
        elif names:
            info["classes"] = [str(name) for name in names]

        source_model = Path(str(train_args.get("model") or "")).stem
        match = re.match(r"(?i)^yolo(\d+)([nsmlx])", source_model)
        if match:
            info["architecture"] = f"YOLOv{match.group(1)}{match.group(2)}-seg"

        info["task"] = getattr(model, "task", None) or "segment"
        info["imgsz"] = int(train_args.get("imgsz") or YOLO_IMGSZ)
        epochs = train_args.get("epochs")
        info["epochs"] = int(epochs) if epochs is not None else None
        info["trained_at"] = checkpoint.get("date")

        raw_metrics = checkpoint.get("train_metrics") or {}

        def metric(name: str) -> float | None:
            value = raw_metrics.get(f"metrics/{name}")
            if isinstance(value, (int, float)):
                return round(float(value), 5)
            return None

        if raw_metrics:
            info["metrics"] = {
                "box": {
                    "precision": metric("precision(B)"),
                    "recall": metric("recall(B)"),
                    "map50": metric("mAP50(B)"),
                    "map50_95": metric("mAP50-95(B)"),
                },
                "mask": {
                    "precision": metric("precision(M)"),
                    "recall": metric("recall(M)"),
                    "map50": metric("mAP50(M)"),
                    "map50_95": metric("mAP50-95(M)"),
                },
            }
        return info

    # ------------------------------------------------------------------
    # Inferensi
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        self._refresh_path()
        info = self.model_info()
        if self.weights_path is None:
            note = (
                "Bobot YOLOv11-seg hasil training belum ditemukan. Letakkan "
                "best.pt di folder models/ untuk mengaktifkan mode YOLO."
            )
        elif not self.runtime_available:
            note = (
                "Bobot ditemukan, tetapi paket ultralytics belum terpasang. "
                "Jalankan: pip install ultralytics"
            )
        else:
            note = f"Model siap: {self.weights_path.name}"
        return {
            "available": self.available,
            "runtime_available": self.runtime_available,
            "weights": self.weights_path.name if self.weights_path else None,
            "note": note,
            "imgsz": YOLO_IMGSZ,
            "model": info,
        }

    def _ensure_model(self) -> None:
        self._refresh_path()
        if self.weights_path is None:
            raise RuntimeError(self.describe()["note"])
        signature = self._signature(self.weights_path)
        if self._model is not None and signature == self._model_signature:
            return

        from ultralytics import YOLO

        try:
            model = YOLO(str(self.weights_path))
        except Exception as exc:  # pragma: no cover - bergantung runtime
            self._load_error = f"Gagal memuat model YOLO: {exc}"
            raise RuntimeError(self._load_error) from exc
        self._model = model
        self._model_signature = signature
        self._load_error = None

    def segment(
        self, image: np.ndarray, conf: float, iou: float
    ) -> list[Instance]:
        if not self.available:
            raise RuntimeError(self.describe()["note"])

        with self._lock:
            self._ensure_model()
            results = self._model.predict(
                source=image,
                conf=float(conf),
                iou=float(iou),
                imgsz=YOLO_IMGSZ,
                task="segment",
                verbose=False,
            )

        if not results:
            return []
        result = results[0]
        if result.masks is None or result.masks.xy is None:
            return []

        height, width = image.shape[:2]
        if result.boxes is not None and result.boxes.conf is not None:
            confidences = result.boxes.conf.cpu().numpy().tolist()
        else:
            confidences = [None] * len(result.masks.xy)

        return masks_from_polygons(
            list(result.masks.xy), height, width, confidences
        )



# ---------------------------------------------------------------------------
# Pengukuran daun dengan OpenCV
# ---------------------------------------------------------------------------


def _straight_edge_line(
    contour: np.ndarray, tolerance: float = 2.0
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Cari sisi lurus terpanjang pada kontur daun.

    Sisi lurus yang panjang biasanya berasal dari objek penghalang, misalnya
    penggaris logam yang diletakkan di atas daun. Fungsi mengembalikan panjang
    sisi, satu titik pada garis, dan arah garis.
    """
    points = contour.reshape(-1, 2).astype(np.float32)
    total = len(points)
    if total < 6:
        return None

    # Sisi lurus sempurna dapat terkompresi menjadi satu segmen kontur saja.
    following = np.roll(points, -1, axis=0)
    edge_lengths = np.hypot(
        following[:, 0] - points[:, 0], following[:, 1] - points[:, 1]
    )
    edge_index = int(edge_lengths.argmax())
    best: tuple[float, int, int] = (
        float(edge_lengths[edge_index]),
        edge_index,
        (edge_index + 1) % total,
    )

    stride = max(1, total // 100)
    indices = list(range(0, total, stride))
    for pos, start_index in enumerate(indices):
        start = points[start_index]
        for end_index in indices[pos + 3 :]:
            end = points[end_index]
            chord = end - start
            chord_length = float(math.hypot(float(chord[0]), float(chord[1])))
            if chord_length < 12.0 or chord_length <= best[0]:
                continue
            normal = np.array([-chord[1], chord[0]], np.float32) / chord_length
            middle = points[start_index + 1 : end_index]
            if middle.size == 0:
                continue
            deviation = float(np.abs((middle - start) @ normal).max())
            if deviation <= tolerance:
                best = (chord_length, start_index, end_index)

    length, start_index, end_index = best
    if end_index > start_index:
        segment = points[start_index : end_index + 1]
    else:
        segment = np.vstack([points[start_index:], points[: end_index + 1]])
    if len(segment) < 2:
        return None
    line = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    point = np.array([line[2], line[3]], np.float32)
    direction = np.array([line[0], line[1]], np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    return length, point, direction / norm


def _reflect_mask(
    mask: np.ndarray, point: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Cerminkan mask terhadap garis (point, direction)."""
    unit = direction / (np.linalg.norm(direction) + 1e-9)
    rotation = 2.0 * np.outer(unit, unit) - np.eye(2)
    offset = point - rotation @ point
    matrix = np.hstack([rotation, offset.reshape(2, 1)]).astype(np.float32)
    return cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    )


def _effective_width(
    mask_u8: np.ndarray,
    contour: np.ndarray,
    length_px: float,
    visible_width_px: float,
) -> tuple[float, bool]:
    """Lebar efektif daun, dengan rekonstruksi cermin bila daun terpotong.

    Bila sisi lurus sepanjang minimal 70% panjang daun terdeteksi, daun
    dianggap terhalang objek lurus (umumnya penggaris). Bagian yang tertutup
    diperkirakan dengan mencerminkan sisi yang terlihat terhadap garis
    penghalang, lalu lebar diukur dari gabungannya.
    """
    edge = _straight_edge_line(contour)
    if edge is None:
        return visible_width_px, False

    edge_length, point, direction = edge
    if edge_length < 0.70 * length_px:
        return visible_width_px, False

    mirrored = _reflect_mask(mask_u8, point, direction)
    union = cv2.bitwise_or(mask_u8, mirrored)
    count, labels, _, _ = cv2.connectedComponentsWithStats(union, 8)
    if count <= 1:
        return visible_width_px, False

    overlap = np.bincount(labels[mask_u8 > 0].ravel(), minlength=count)
    overlap[0] = 0
    keep = int(overlap.argmax())
    if keep == 0:
        return visible_width_px, False

    union = (labels == keep).astype(np.uint8) * 255
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return visible_width_px, False

    completed_contour = max(contours, key=cv2.contourArea)
    side_a, side_b = cv2.minAreaRect(completed_contour)[1]
    completed_width = float(min(side_a, side_b))

    # Batas kewajaran: rekonstruksi tidak boleh melebihi dua kali lebar terlihat.
    if not (1.10 * visible_width_px <= completed_width <= 2.15 * visible_width_px):
        return visible_width_px, False
    return completed_width, True


def measure_instances(
    instances: list[Instance], pixels_per_cm: float | None = None
) -> list[Leaf]:
    height, width = instances[0].mask.shape[:2] if instances else (0, 0)
    image_area = float(height * width)
    min_area = max(160.0, image_area * MIN_LEAF_AREA_RATIO)
    max_area = image_area * MAX_LEAF_AREA_RATIO

    prepared: list[dict[str, Any]] = []
    for instance in instances:
        mask_u8 = (instance.mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area_px = float(instance.mask.sum())
        if area_px < min_area or area_px > max_area:
            continue
        rect = cv2.minAreaRect(contour)
        side_a, side_b = rect[1]
        length_px = float(max(side_a, side_b))
        visible_width_px = float(min(side_a, side_b))
        if length_px < 6 or visible_width_px < 3:
            continue
        leaf_width_px, occluded = _effective_width(
            mask_u8, contour, length_px, visible_width_px
        )
        ratio = length_px / leaf_width_px if leaf_width_px > 0 else 0.0
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            centroid = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
        else:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w / 2, y + h / 2)
        prepared.append(
            {
                "instance": instance,
                "contour": contour,
                "rect": rect,
                "length_px": length_px,
                "width_px": leaf_width_px,
                "width_visible_px": visible_width_px,
                "occluded": occluded,
                "area_px": area_px,
                "ratio": ratio,
                "centroid": centroid,
            }
        )

    # Urutkan kiri→kanan, atas→bawah agar penomoran stabil dan mudah dibaca.
    prepared.sort(key=lambda item: (round(item["centroid"][0] / 40), item["centroid"][1]))
    if len(prepared) > MAX_LEAVES:
        # Batas diterapkan pada instance terbesar, lalu diurutkan ulang secara
        # spasial supaya penomoran tetap konsisten.
        prepared = sorted(
            prepared, key=lambda item: item["area_px"], reverse=True
        )[:MAX_LEAVES]
        prepared.sort(
            key=lambda item: (round(item["centroid"][0] / 40), item["centroid"][1])
        )

    leaves: list[Leaf] = []
    for index, item in enumerate(prepared, start=1):
        color_index = (index - 1) % len(PALETTE_BGR)
        leaves.append(
            Leaf(
                index=index,
                mask=item["instance"].mask,
                contour=item["contour"],
                rect=item["rect"],
                length_px=item["length_px"],
                width_px=item["width_px"],
                width_visible_px=item["width_visible_px"],
                occluded=item["occluded"],
                area_px=item["area_px"],
                ratio=item["ratio"],
                confidence=item["instance"].confidence,
                centroid=item["centroid"],
                color_bgr=PALETTE_BGR[color_index],
                color_hex=PALETTE_HEX[color_index],
            )
        )
    return leaves


def aggregate_features(
    leaves: list[Leaf], image_shape: tuple[int, int]
) -> dict[str, float]:
    height, width = image_shape
    image_area = float(height * width)
    diagonal = math.hypot(width, height)

    if not leaves:
        return {
            "leaf_count": 0,
            "mean_length": 0.0,
            "mean_width": 0.0,
            "max_length": 0.0,
            "max_width": 0.0,
            "total_area": 0.0,
            "mean_ratio": 0.0,
            "total_area_ratio": 0.0,
            "mean_area_ratio": 0.0,
            "mean_length_ratio": 0.0,
            "max_length_ratio": 0.0,
        }

    lengths = [leaf.length_px for leaf in leaves]
    widths = [leaf.width_px for leaf in leaves]
    areas = [leaf.area_px for leaf in leaves]
    ratios = [leaf.ratio for leaf in leaves]

    total_area = float(sum(areas))
    mean_length = float(np.mean(lengths))
    max_length = float(np.max(lengths))

    return {
        "leaf_count": int(len(leaves)),
        "mean_length": mean_length,
        "mean_width": float(np.mean(widths)),
        "max_length": max_length,
        "max_width": float(np.max(widths)),
        "total_area": total_area,
        "mean_ratio": float(np.mean(ratios)),
        "total_area_ratio": total_area / image_area if image_area else 0.0,
        "mean_area_ratio": (total_area / len(leaves)) / image_area if image_area else 0.0,
        "mean_length_ratio": mean_length / diagonal if diagonal else 0.0,
        "max_length_ratio": max_length / diagonal if diagonal else 0.0,
    }


def feature_vector(aggregated: dict[str, float]) -> dict[str, float]:
    return {column: float(aggregated.get(column, 0.0)) for column in FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------


def draw_annotations(image: np.ndarray, leaves: list[Leaf]) -> np.ndarray:
    canvas = image.copy()
    if not leaves:
        _draw_message(canvas, "Tidak ada daun terdeteksi")
        return canvas

    overlay = image.copy()
    for leaf in leaves:
        overlay[leaf.mask] = leaf.color_bgr
    canvas = cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.38, min(0.62, image.shape[1] / 1100))
    thickness = max(1, int(round(scale * 2)))

    for leaf in leaves:
        cv2.drawContours(canvas, [leaf.contour], -1, leaf.color_bgr, 2, cv2.LINE_AA)
        box = cv2.boxPoints(leaf.rect)
        box = np.intp(box)
        cv2.polylines(canvas, [box], True, (255, 255, 255), 1, cv2.LINE_AA)

        label = f"D{leaf.index}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        center_x = int(leaf.centroid[0])
        center_y = int(leaf.centroid[1])
        left = center_x - text_width // 2
        top = center_y - text_height // 2
        pad = max(3, int(round(scale * 7)))
        cv2.rectangle(
            canvas,
            (left - pad, top - pad),
            (left + text_width + pad, top + text_height + baseline + pad),
            (17, 32, 24),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (left, top + text_height),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    _draw_legend(canvas, len(leaves))
    return canvas


def _draw_message(canvas: np.ndarray, message: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(0.85, canvas.shape[1] / 900))
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        message, font, scale, thickness
    )
    x = max(12, (canvas.shape[1] - text_width) // 2)
    y = max(30, canvas.shape[0] // 2)
    cv2.rectangle(
        canvas,
        (x - 14, y - text_height - 14),
        (x + text_width + 14, y + baseline + 14),
        (17, 32, 24),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_legend(canvas: np.ndarray, leaf_count: int) -> None:
    text = f"{leaf_count} instance daun"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    cv2.rectangle(
        canvas,
        (12, 12),
        (12 + text_width + 18, 12 + text_height + baseline + 14),
        (17, 32, 24),
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (21, 12 + text_height + 6),
        font,
        scale,
        (214, 244, 198),
        thickness,
        cv2.LINE_AA,
    )


def mask_thumbnail(leaf: Leaf, size: int = 112) -> str:
    """Thumbnail mask yang dipotong ke bounding box agar daun terlihat jelas."""
    padding = 10
    inner = size - padding * 2

    ys, xs = np.nonzero(leaf.mask)
    if len(xs) == 0:
        crop = leaf.mask.astype(np.uint8) * 255
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = leaf.mask[y0:y1, x0:x1].astype(np.uint8) * 255

    height, width = crop.shape[:2]
    scale = min(inner / max(height, width), 1.0)
    resized = cv2.resize(
        crop,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:] = (18, 30, 23)
    y_offset = (size - resized.shape[0]) // 2
    x_offset = (size - resized.shape[1]) // 2
    region = canvas[
        y_offset : y_offset + resized.shape[0],
        x_offset : x_offset + resized.shape[1],
    ]
    region[resized > 0] = leaf.color_bgr
    contours, _ = cv2.findContours(
        resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shifted = [contour + np.array([[x_offset, y_offset]]) for contour in contours]
    cv2.drawContours(canvas, shifted, -1, (245, 250, 245), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"D{leaf.index}",
        (6, size - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (210, 230, 210),
        1,
        cv2.LINE_AA,
    )
    return encode_data_url(canvas, mime="image/png")


# ---------------------------------------------------------------------------
# Orkestrasi pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self) -> None:
        self.yolo = YoloSegmenter()
        self.classifier = PhaseClassifier()

    def status(self) -> dict[str, Any]:
        yolo_info = self.yolo.describe()
        classifier_info = self.classifier.describe()
        return {
            "yolo": yolo_info,
            "classifier": classifier_info,
            "inference_size": INFER_MAX_SIZE,
            "phase_age": PHASE_TO_AGE,
        }

    def _segment(
        self,
        detect_image: np.ndarray,
        classic_image: np.ndarray,
        requested: str,
        conf: float,
        iou: float,
    ) -> SegmentResult:
        result = SegmentResult()
        requested = requested if requested in {"auto", "yolo", "classic"} else "auto"

        if requested in {"auto", "yolo"}:
            if self.yolo.available:
                try:
                    instances = self.yolo.segment(
                        detect_image, conf=conf, iou=iou
                    )
                    if instances:
                        model_info = self.yolo.model_info() or {}
                        architecture = (
                            model_info.get("architecture") or "YOLOv11-seg"
                        )
                        mask_map50 = (model_info.get("metrics") or {}).get(
                            "mask", {}
                        ).get("map50")
                        metric_note = (
                            f", mAP50 mask {mask_map50:.3f}".replace(".", ",")
                            if mask_map50 is not None
                            else ""
                        )
                        result.instances = instances
                        result.engine = "yolo"
                        result.label = architecture
                        result.note = (
                            f"Inferensi memakai bobot {self.yolo.weights_path.name}"
                            f"{metric_note}. Citra asli setelah resize 640 px dipakai "
                            "langsung, tanpa CLAHE, agar sesuai distribusi data latih."
                        )
                        return result
                    result.warnings.append(
                        "YOLOv11-seg tidak menemukan instance daun pada foto ini."
                    )
                except Exception as exc:  # pragma: no cover - bergantung runtime
                    result.warnings.append(f"Inferensi YOLO gagal: {exc}")
            else:
                result.warnings.append(self.yolo.describe()["note"])

            if requested == "yolo":
                result.engine = "yolo"
                result.label = "YOLOv11-seg"
                result.note = "Tidak ada instance daun yang dihasilkan model."
                return result

        result.instances = classic_segment(classic_image)
        result.engine = "classic"
        result.label = "Segmentasi Klasik (demo)"
        result.note = (
            "Mode demo: daun disegmentasi dengan HSV + Excess Green + watershed. "
            "Pasang bobot YOLOv11-seg terlatih di folder models/ untuk memakai model utama."
        )
        return result

    def analyze(
        self,
        data: bytes,
        filename: str,
        conf: float = 0.25,
        iou: float = 0.50,
        pixels_per_cm: float | None = None,
        engine: str = "auto",
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        started = time.perf_counter()

        original = decode_image(data)
        resized, preprocessed = preprocess_image(original)
        timings["preprocess_ms"] = round((time.perf_counter() - started) * 1000, 1)

        segment_started = time.perf_counter()
        segmentation = self._segment(
            resized,
            preprocessed,
            requested=engine,
            conf=max(0.01, min(conf, 0.99)),
            iou=max(0.05, min(iou, 0.95)),
        )
        timings["segment_ms"] = round(
            (time.perf_counter() - segment_started) * 1000, 1
        )

        feature_started = time.perf_counter()
        leaves = measure_instances(segmentation.instances, pixels_per_cm=pixels_per_cm)
        aggregated = aggregate_features(leaves, preprocessed.shape[:2])
        timings["features_ms"] = round(
            (time.perf_counter() - feature_started) * 1000, 1
        )

        classifier_started = time.perf_counter()
        prediction = self.classifier.predict(
            feature_vector(aggregated),
            context={
                "total_area_ratio": aggregated["total_area_ratio"],
                "mean_area_ratio": aggregated["mean_area_ratio"],
                "mean_length_ratio": aggregated["mean_length_ratio"],
                "max_length_ratio": aggregated["max_length_ratio"],
            },
        )
        timings["classify_ms"] = round(
            (time.perf_counter() - classifier_started) * 1000, 1
        )

        recommendations = build_recommendations(
            phase=prediction.get("phase"),
            features=aggregated,
            context={
                "total_area_ratio": aggregated["total_area_ratio"],
                "mean_length_ratio": aggregated["mean_length_ratio"],
            },
            confidence=float(prediction.get("confidence") or 0.0),
            is_demo=bool(prediction.get("is_demo")),
            leaf_count=int(aggregated["leaf_count"]),
        )

        annotated = draw_annotations(resized, leaves)

        warnings = list(segmentation.warnings)
        if not leaves:
            warnings.append(
                "Tidak ada daun yang lolos filter area minimum. Coba foto yang lebih "
                "terang, dekat, atau naikkan sensitivitas deteksi."
            )
        if len(leaves) >= MAX_LEAVES:
            warnings.append(
                f"Instance dibatasi {MAX_LEAVES} daun terbesar. Pastikan foto hanya "
                "memuat satu tanaman agar agregasi fitur valid."
            )
        occluded_leaves = [leaf.index for leaf in leaves if leaf.occluded]
        if occluded_leaves:
            labels = ", ".join(f"D{index}" for index in occluded_leaves)
            warnings.append(
                f"Lebar {labels} direkonstruksi karena daun terpotong objek lurus "
                "(misalnya penggaris). Nilai luas dan panjang tetap dihitung dari "
                "piksel yang terlihat."
            )

        units = {
            "calibrated": pixels_per_cm is not None,
            "pixels_per_cm": pixels_per_cm,
            "length": "cm" if pixels_per_cm else "px",
            "area": "cm²" if pixels_per_cm else "px²",
        }

        def convert_length(value_px: float) -> float:
            return round(value_px / pixels_per_cm, 2) if pixels_per_cm else round(value_px, 1)

        def convert_area(value_px: float) -> float:
            if not pixels_per_cm:
                return round(value_px, 1)
            return round(value_px / (pixels_per_cm**2), 2)

        leaf_rows = []
        for leaf in leaves:
            leaf_rows.append(
                {
                    "id": leaf.index,
                    "color": leaf.color_hex,
                    "length": convert_length(leaf.length_px),
                    "width": convert_length(leaf.width_px),
                    "width_visible": convert_length(leaf.width_visible_px),
                    "occluded": bool(leaf.occluded),
                    "area": convert_area(leaf.area_px),
                    "ratio": round(leaf.ratio, 2),
                    "confidence": (
                        round(float(leaf.confidence), 3)
                        if leaf.confidence is not None
                        else None
                    ),
                    "centroid": [round(leaf.centroid[0], 1), round(leaf.centroid[1], 1)],
                    "mask": mask_thumbnail(leaf),
                }
            )

        aggregated_public = {
            "leaf_count": int(aggregated["leaf_count"]),
            "occluded_count": sum(1 for leaf in leaves if leaf.occluded),
            "mean_length": convert_length(aggregated["mean_length"]),
            "mean_width": convert_length(aggregated["mean_width"]),
            "max_length": convert_length(aggregated["max_length"]),
            "max_width": convert_length(aggregated["max_width"]),
            "total_area": convert_area(aggregated["total_area"]),
            "mean_ratio": round(aggregated["mean_ratio"], 2),
            "total_area_ratio": round(aggregated["total_area_ratio"] * 100, 2),
            "mean_area_ratio": round(aggregated["mean_area_ratio"] * 100, 3),
            "mean_length_ratio": round(aggregated["mean_length_ratio"] * 100, 2),
        }

        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)

        return {
            "image": {
                "filename": filename,
                "width": int(preprocessed.shape[1]),
                "height": int(preprocessed.shape[0]),
                "original": encode_data_url(resized),
                "preprocessed": encode_data_url(preprocessed),
                "annotated": encode_data_url(annotated),
            },
            "engine": {
                "requested": engine,
                "used": segmentation.engine,
                "label": segmentation.label,
                "is_demo": segmentation.engine != "yolo",
                "note": segmentation.note,
                "yolo": self.yolo.describe(),
            },
            "leaves": leaf_rows,
            "aggregated": aggregated_public,
            "aggregated_raw": {key: float(value) for key, value in aggregated.items()},
            "prediction": prediction,
            "recommendations": recommendations,
            "timings": timings,
            "units": units,
            "warnings": warnings,
        }
