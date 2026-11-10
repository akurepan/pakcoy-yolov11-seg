"""Prediksi fase pertumbuhan: TFLite → scikit-learn → heuristik demo."""

from __future__ import annotations

import importlib.util
import json
import pickle
from typing import Any

import numpy as np

from .config import (
    FEATURE_COLUMNS,
    LABEL_ENCODER_CANDIDATES,
    MODEL_INFO_FILE,
    PHASE_TO_AGE,
    PHASES,
    SCALER_CANDIDATES,
    SKLEARN_CANDIDATES,
    TFLITE_CANDIDATES,
    first_existing,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _softmax_from_distance(maturity: float) -> dict[str, float]:
    centers = [0.16, 0.50, 0.84]
    sigma = 0.18
    scores = [
        float(np.exp(-((maturity - center) ** 2) / (2 * sigma**2)))
        for center in centers
    ]
    total = sum(scores) or 1.0
    return {phase: score / total for phase, score in zip(PHASES, scores)}


class PhaseClassifier:
    """Memuat model terlatih bila tersedia, jika tidak memakai aturan demo."""

    def __init__(self) -> None:
        self.kind = "heuristic"
        self.label = "Aturan heuristik (demo)"
        self.model_file: str | None = None
        self.load_notes: list[str] = []
        self._interpreter = None
        self._scaler = None
        self._sklearn_model = None
        self._labels: list[str] = list(PHASES)
        self._model_info = self._read_model_info()
        self._load_models()

    # ------------------------------------------------------------------
    # Pemuatan model
    # ------------------------------------------------------------------

    @staticmethod
    def _read_model_info() -> dict[str, Any]:
        if not MODEL_INFO_FILE.exists():
            return {}
        try:
            return json.loads(MODEL_INFO_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _load_models(self) -> None:
        if self._try_tflite():
            return
        if self._try_sklearn():
            return
        self.kind = "heuristic"
        self.label = "Aturan heuristik (demo)"
        self.model_file = None

    def _try_tflite(self) -> bool:
        tflite_path = first_existing(TFLITE_CANDIDATES)
        scaler_path = first_existing(SCALER_CANDIDATES)
        encoder_path = first_existing(LABEL_ENCODER_CANDIDATES)
        if not (tflite_path and scaler_path and encoder_path):
            return False

        has_tensorflow = importlib.util.find_spec("tensorflow") is not None
        has_runtime = importlib.util.find_spec("tflite_runtime") is not None
        if not (has_tensorflow or has_runtime):
            self.load_notes.append(
                "Model TFLite ditemukan, tetapi TensorFlow/tflite-runtime belum "
                "terpasang. Jalankan: pip install tensorflow"
            )
            return False

        try:
            if has_tensorflow:
                import tensorflow as tf

                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            else:  # pragma: no cover - opsi runtime alternatif
                from tflite_runtime.interpreter import Interpreter

                interpreter = Interpreter(model_path=str(tflite_path))

            interpreter.allocate_tensors()
            self._interpreter = interpreter
            self._input_detail = interpreter.get_input_details()[0]
            self._output_detail = interpreter.get_output_details()[0]

            with open(scaler_path, "rb") as handle:
                self._scaler = pickle.load(handle)
            with open(encoder_path, "rb") as handle:
                encoder = pickle.load(handle)
            self._labels = [str(label) for label in encoder.classes_]
            self.kind = "tflite"
            self.label = "MLP · TensorFlow Lite"
            self.model_file = tflite_path.name
            return True
        except Exception as exc:  # pragma: no cover - bergantung runtime
            self.load_notes.append(f"Gagal memuat model TFLite: {exc}")
            return False

    def _try_sklearn(self) -> bool:
        model_path = first_existing(SKLEARN_CANDIDATES)
        if model_path is None:
            return False
        try:
            with open(model_path, "rb") as handle:
                model = pickle.load(handle)
            encoder_path = first_existing(LABEL_ENCODER_CANDIDATES)
            if encoder_path is not None:
                with open(encoder_path, "rb") as handle:
                    encoder = pickle.load(handle)
                labels = [str(label) for label in encoder.classes_]
            else:
                raw_labels = list(getattr(model, "classes_", []))
                if not raw_labels or not all(
                    isinstance(label, (str, np.str_)) for label in raw_labels
                ):
                    self.load_notes.append(
                        "Model scikit-learn ditemukan tanpa label_encoder.pkl; "
                        "label kelas tidak dapat dipetakan."
                    )
                    return False
                labels = [str(label) for label in raw_labels]

            self._sklearn_model = model
            self._labels = labels
            self.kind = "sklearn"
            self.label = (
                "Random Forest · baseline"
                if "rf" in model_path.name.lower()
                else "MLP · scikit-learn"
            )
            self.model_file = model_path.name
            return True
        except Exception as exc:
            self.load_notes.append(f"Gagal memuat model scikit-learn: {exc}")
            return False

    # ------------------------------------------------------------------
    # Prediksi
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        trained = self.kind in {"tflite", "sklearn"}
        if trained:
            detail = (
                f"Model klasifikasi aktif: {self.label} ({self.model_file}). "
                "Fitur masukan dihitung dari instance mask pada citra 640 px."
            )
        else:
            detail = (
                "Belum ada model klasifikasi terlatih. Prediksi memakai aturan "
                "heuristik berbasis jumlah daun, tutupan area, dan rasio bentuk. "
                "Letakkan pakcoy_mlp.tflite + feature_scaler.pkl + label_encoder.pkl "
                "atau best_rf.pkl + label_encoder.pkl di folder models/."
            )
        return {
            "kind": self.kind,
            "label": self.label,
            "is_demo": not trained,
            "model_file": self.model_file,
            "classes": self._labels,
            "detail": detail,
            "notes": self.load_notes,
            "model_info": self._model_info,
        }

    def predict(
        self, features: dict[str, float], context: dict[str, float]
    ) -> dict[str, Any]:
        if int(features.get("leaf_count", 0)) == 0:
            return {
                "phase": None,
                "phase_index": None,
                "confidence": 0.0,
                "probabilities": {phase: 0.0 for phase in PHASES},
                "estimated_age": "-",
                "model": self.label,
                "is_demo": self.kind == "heuristic",
                "source": self.kind,
                "note": "Tidak ada daun terdeteksi sehingga fase belum dapat ditentukan.",
                "low_confidence": True,
            }

        vector = np.array(
            [[float(features.get(column, 0.0)) for column in FEATURE_COLUMNS]],
            dtype=np.float32,
        )

        if self.kind == "tflite" and self._interpreter is not None:
            return self._predict_tflite(vector)
        if self.kind == "sklearn" and self._sklearn_model is not None:
            return self._predict_sklearn(vector)
        return self._predict_heuristic(features, context)

    def _predict_tflite(self, vector: np.ndarray) -> dict[str, Any]:
        scaled = self._scaler.transform(vector).astype(np.float32)
        self._interpreter.set_tensor(self._input_detail["index"], scaled)
        self._interpreter.invoke()
        probabilities = np.asarray(
            self._interpreter.get_tensor(self._output_detail["index"])[0],
            dtype=np.float64,
        )
        return self._format(
            probabilities,
            source="tflite",
            model=self.label,
            note="Prediksi dari model MLP yang dikonversi ke TensorFlow Lite.",
        )

    def _predict_sklearn(self, vector: np.ndarray) -> dict[str, Any]:
        if hasattr(self._sklearn_model, "predict_proba"):
            probabilities = np.asarray(
                self._sklearn_model.predict_proba(vector)[0], dtype=np.float64
            )
        else:  # pragma: no cover - model tanpa predict_proba
            prediction = self._sklearn_model.predict(vector)[0]
            probabilities = np.zeros(len(self._labels), dtype=np.float64)
            probabilities[self._labels.index(str(prediction))] = 1.0
        return self._format(
            probabilities,
            source="sklearn",
            model=self.label,
            note="Prediksi dari model klasifikasi scikit-learn terlatih.",
        )

    def _format(
        self,
        probabilities: np.ndarray,
        source: str,
        model: str,
        note: str,
    ) -> dict[str, Any]:
        mapped = {phase: 0.0 for phase in PHASES}
        for label, probability in zip(self._labels, probabilities):
            if label in mapped:
                mapped[label] += float(probability)
        total = sum(mapped.values())
        if total > 0:
            mapped = {phase: value / total for phase, value in mapped.items()}
        else:
            mapped = {phase: 1.0 / len(PHASES) for phase in PHASES}

        phase = max(mapped, key=mapped.get)
        confidence = float(mapped[phase])
        return {
            "phase": phase,
            "phase_index": PHASES.index(phase),
            "confidence": confidence,
            "probabilities": mapped,
            "estimated_age": PHASE_TO_AGE[phase],
            "model": model,
            "is_demo": False,
            "source": source,
            "note": note,
            "low_confidence": confidence < 0.55,
        }

    def _predict_heuristic(
        self, features: dict[str, float], context: dict[str, float]
    ) -> dict[str, Any]:
        leaf_count = float(features.get("leaf_count", 0))
        total_area_ratio = float(context.get("total_area_ratio", 0.0))
        mean_length_ratio = float(context.get("mean_length_ratio", 0.0))

        count_score = _clamp((leaf_count - 1.0) / 8.0)
        area_score = _clamp(total_area_ratio / 0.18)
        length_score = _clamp((mean_length_ratio - 0.05) / 0.32)

        maturity = (
            0.36 * count_score + 0.44 * area_score + 0.20 * length_score
        )
        probabilities = _softmax_from_distance(maturity)

        phase = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[phase])
        return {
            "phase": phase,
            "phase_index": PHASES.index(phase),
            "confidence": confidence,
            "probabilities": probabilities,
            "estimated_age": PHASE_TO_AGE[phase],
            "model": "Aturan heuristik (demo)",
            "is_demo": True,
            "source": "heuristic",
            "note": (
                "Prediksi demo berbasis aturan, bukan model terlatih. Skor "
                f"kematangan = {maturity:.2f} (0 = semai, 1 = panen)."
            ),
            "low_confidence": confidence < 0.55,
            "maturity_score": round(maturity, 3),
        }
