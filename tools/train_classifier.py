"""Latih classifier fase pakcoy dari CSV fitur dan simpan artefaknya.

Contoh:

    .venv\\Scripts\\python tools\\train_classifier.py --csv fitur-pakcoy.csv --model rf
    .venv\\Scripts\\python tools\\train_classifier.py --csv fitur-pakcoy.csv --model mlp ^
        --metrics-out hasil\\klasifikasi-metrik.json

Keluaran di folder `models/`:

- `best_rf.pkl` (Random Forest) atau `best_mlp_sklearn.pkl` (MLP)
- `label_encoder.pkl`

Keduanya disimpan dengan `pickle`, karena aplikasi memuatnya dengan
`pickle.load`. MLP dibungkus `StandardScaler` di dalam `Pipeline` supaya
inferensi memakai skala yang sama dengan training.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import FEATURE_COLUMNS, PHASES  # noqa: E402

MODEL_FILENAMES = {
    "rf": "best_rf.pkl",
    "mlp": "best_mlp_sklearn.pkl",
}


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"CSV tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in fieldnames]
        if missing:
            raise SystemExit(
                "Kolom wajib tidak ditemukan: " + ", ".join(missing) + "\n"
                "Gunakan tools/build_feature_csv.py untuk membuat CSV yang benar."
            )
        rows = list(reader)

    if not rows:
        raise SystemExit("CSV kosong.")

    try:
        features = np.asarray(
            [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Ada nilai fitur yang bukan angka: {exc}") from exc

    labels = np.asarray([str(row["label"]).strip() for row in rows], dtype=object)

    if not np.isfinite(features).all():
        raise SystemExit("Fitur mengandung NaN atau infinity. Periksa CSV.")
    unknown = sorted(set(labels) - set(PHASES))
    if unknown:
        raise SystemExit(
            "Label tidak dikenal: "
            + ", ".join(unknown)
            + "\nLabel yang valid: "
            + ", ".join(PHASES)
        )
    return features, labels


def build_estimator(name: str, random_state: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler

    return SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=3000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Latih classifier fase pakcoy dan simpan artefak untuk aplikasi."
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV fitur + label.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FILENAMES),
        default="rf",
        help="Jenis model (default: rf).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models",
        help="Folder keluaran artefak (default: models).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data uji; 0 untuk memakai semua data sebagai latih.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        help="Simpan metrik evaluasi sebagai JSON (opsional).",
    )
    args = parser.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    features, labels = load_dataset(args.csv)
    counts = Counter(labels.tolist())
    print(f"Sampel: {len(labels)}")
    for phase in PHASES:
        print(f"  {phase:12} {counts.get(phase, 0):4}")
    rare = [phase for phase in PHASES if counts.get(phase, 0) < 15]
    if rare:
        print(
            "\nPeringatan: kelas <15 sampel: "
            + ", ".join(rare)
            + ".\nHasil evaluasi belum layak diklaim sebagai akurasi penelitian."
        )
    if args.model == "mlp" and len(labels) < 50:
        print("Peringatan: MLP biasanya butuh lebih dari 50 sampel agar stabil.")

    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)

    can_split = (
        args.test_size > 0
        and len(labels) >= 20
        and min(counts.values()) >= 3
        and len(encoder.classes_) >= 2
    )
    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            encoded,
            test_size=args.test_size,
            stratify=encoded,
            random_state=args.random_state,
        )
    else:
        x_train, y_train = features, encoded
        x_test = y_test = None
        print(
            "Info: data belum cukup untuk pembagian uji berstrata; "
            "model dilatih pada seluruh data."
        )

    estimator = build_estimator(args.model, args.random_state)
    estimator.fit(x_train, y_train)

    metrics: dict[str, object] = {
        "model": args.model,
        "features": FEATURE_COLUMNS,
        "classes": [str(label) for label in encoder.classes_],
        "n_samples": int(len(labels)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)) if y_test is not None else 0,
        "class_counts": {phase: int(counts.get(phase, 0)) for phase in PHASES},
    }

    if x_test is not None and y_test is not None:
        prediction = estimator.predict(x_test)
        report = classification_report(
            y_test,
            prediction,
            labels=list(range(len(encoder.classes_))),
            target_names=[str(label) for label in encoder.classes_],
            zero_division=0,
            output_dict=True,
        )
        matrix = confusion_matrix(
            y_test, prediction, labels=list(range(len(encoder.classes_)))
        )
        accuracy = float((prediction == y_test).mean())
        metrics["accuracy"] = accuracy
        metrics["classification_report"] = report
        metrics["confusion_matrix"] = matrix.tolist()

        print(f"\nAkurasi uji: {accuracy:.4f} ({len(y_test)} sampel)")
        print(
            classification_report(
                y_test,
                prediction,
                labels=list(range(len(encoder.classes_))),
                target_names=[str(label) for label in encoder.classes_],
                zero_division=0,
            )
        )
        print("Confusion matrix (baris = aktual, kolom = prediksi):")
        print(matrix)

    if min(counts.values()) >= 2:
        folds = min(5, min(counts.values()))
        if folds >= 2:
            scores = cross_val_score(
                build_estimator(args.model, args.random_state),
                features,
                encoded,
                cv=StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=args.random_state
                ),
                scoring="accuracy",
            )
            metrics["cv_folds"] = folds
            metrics["cv_mean"] = float(scores.mean())
            metrics["cv_std"] = float(scores.std())
            print(
                f"Cross-validation: {scores.mean():.4f} ± {scores.std():.4f} "
                f"({folds} fold)"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / MODEL_FILENAMES[args.model]
    encoder_path = args.out_dir / "label_encoder.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(estimator, handle)
    with encoder_path.open("wb") as handle:
        pickle.dump(encoder, handle)
    print(f"\nModel disimpan : {model_path}")
    print(f"Encoder disimpan: {encoder_path}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metrik disimpan: {args.metrics_out}")

    if args.out_dir.resolve() == (ROOT / "models").resolve():
        from app.classifier import PhaseClassifier

        loaded = PhaseClassifier().describe()
        print(
            "\nVerifikasi pemuatan: "
            f"kind={loaded['kind']}, label={loaded['label']}, "
            f"is_demo={loaded['is_demo']}"
        )
        if loaded["kind"] == "heuristic":
            print("Gagal memuat artefak. Catatan:")
            for note in loaded["notes"]:
                print(f"  - {note}")
            return 1

    print("\nJalankan ulang server agar classifier baru dipakai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
