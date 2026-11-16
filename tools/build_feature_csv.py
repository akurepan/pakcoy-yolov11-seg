"""Bangun CSV fitur fase dari kumpulan foto.

Skrip menjalankan pipeline yang sama dengan aplikasi (YOLOv11-seg + pengukuran
OpenCV), lalu menulis satu baris per foto:

    leaf_count,mean_length,mean_width,max_length,max_width,total_area,mean_ratio,label

Label dapat diambil dari nama file (pola `day12`, `day_12`, `day-12`) atau dari
CSV pemetaan `file,label`. Contoh:

    .venv\\Scripts\\python tools\\build_feature_csv.py ^
        --images dataset\\train\\images dataset\\valid\\images ^
        --out fitur-pakcoy.csv

    .venv\\Scripts\\python tools\\build_feature_csv.py ^
        --images path\\ke\\foto --labels label.csv --out fitur-pakcoy.csv

Secara bawaan hanya hasil segmentasi YOLO yang dipakai, karena fitur dari
segmentasi klasik punya distribusi berbeda. Tambahkan `--include-classic` bila
memang ingin mencampurnya. Berkas yang namanya memuat "thumb" dilewati agar
thumbnail tidak menduplikasi foto asli.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import FEATURE_COLUMNS, PHASES  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DAY_PATTERN = r"day[-_ ]?(\d{1,3})"


def phase_from_day(day: int) -> str | None:
    if 0 <= day <= 14:
        return "Penyemaian"
    if 15 <= day <= 30:
        return "Vegetatif"
    if 31 <= day <= 45:
        return "Siap Panen"
    return None


def read_label_map(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"file", "label"}.issubset(reader.fieldnames):
            raise SystemExit(
                f"{path} harus punya kolom 'file' dan 'label'. "
                f"Kolom ditemukan: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("file") or "").strip()
            label = (row.get("label") or "").strip()
            if name and label:
                labels[name] = label
    return labels


def collect_images(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            raise SystemExit(f"Folder tidak ditemukan: {directory}")
        files.extend(
            sorted(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bangun CSV fitur untuk melatih classifier fase pakcoy."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Satu atau lebih folder citra (dibaca rekursif).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="CSV opsional berisi kolom file,label.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "fitur-pakcoy.csv",
        help="Lokasi CSV keluaran (default: fitur-pakcoy.csv).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence YOLO.")
    parser.add_argument(
        "--engine",
        choices=["auto", "yolo", "classic"],
        default="auto",
        help="Mesin segmentasi (default: auto).",
    )
    parser.add_argument(
        "--include-classic",
        action="store_true",
        help="Ikutkan hasil segmentasi klasik (tidak disarankan).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Batasi jumlah citra yang diproses (0 = semua).",
    )
    parser.add_argument(
        "--include-thumbs",
        action="store_true",
        help="Ikutkan berkas berlabel 'thumb' (default: dilewati).",
    )
    args = parser.parse_args()

    label_map = read_label_map(args.labels) if args.labels else {}
    files = collect_images([Path(item) for item in args.images])
    if not args.include_thumbs:
        files = [path for path in files if "thumb" not in path.stem.lower()]
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit("Tidak ada citra JPG/PNG/WEBP di folder yang diberikan.")

    pipeline = Pipeline()
    status = pipeline.status()
    if not status["yolo"]["available"]:
        raise SystemExit(
            "Mode YOLO tidak aktif. Letakkan models/best.pt dan pasang "
            "ultralytics (pip install -r requirements-ml.txt)."
        )

    rows: list[dict[str, object]] = []
    skipped: Counter[str] = Counter()

    for path in files:
        label = label_map.get(path.name)
        day: int | None = None
        if label is None:
            match = re.search(DAY_PATTERN, path.stem, flags=re.IGNORECASE)
            if match:
                day = int(match.group(1))
                label = phase_from_day(day)
        if label is None:
            skipped["label tidak diketahui"] += 1
            continue
        if label not in PHASES:
            skipped[f"label tidak dikenal: {label}"] += 1
            continue

        try:
            result = pipeline.analyze(
                path.read_bytes(), path.name, args.conf, 0.50, None, args.engine
            )
        except Exception as exc:  # noqa: BLE001 - laporkan lalu lanjut
            skipped[f"gagal diproses: {type(exc).__name__}"] += 1
            continue

        engine_used = str(result["engine"]["used"])
        if engine_used != "yolo" and not args.include_classic:
            skipped[f"bukan YOLO ({engine_used})"] += 1
            continue

        features = result["aggregated_raw"]
        if int(features.get("leaf_count", 0)) == 0:
            skipped["tidak ada daun"] += 1
            continue

        row: dict[str, object] = {
            column: float(features.get(column, 0.0)) for column in FEATURE_COLUMNS
        }
        row.update(
            {
                "label": label,
                "source": path.name,
                "engine": engine_used,
                "day": day if day is not None else "",
            }
        )
        rows.append(row)

    if not rows:
        print("Tidak ada baris yang bisa ditulis. Rincian lewatan:")
        for reason, count in skipped.most_common():
            print(f"  - {reason}: {count}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FEATURE_COLUMNS + ["label", "source", "engine", "day"]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["label"]) for row in rows)
    print(f"CSV ditulis: {args.out} ({len(rows)} baris)")
    for phase in PHASES:
        print(f"  {phase:12} {counts.get(phase, 0):4} foto")
    if skipped:
        print("Dilewati:")
        for reason, count in skipped.most_common():
            print(f"  - {reason}: {count}")

    rare = [phase for phase in PHASES if counts.get(phase, 0) < 15]
    if rare:
        print(
            "\nPeringatan: kelas dengan <15 foto belum cukup untuk klaim akurasi: "
            + ", ".join(rare)
        )
    print("\nLangkah berikutnya:")
    print(
        "  .venv\\Scripts\\python tools\\train_classifier.py "
        f"--csv \"{args.out}\" --model rf"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
