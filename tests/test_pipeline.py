"""Uji asap pipeline tanpa dependensi tambahan (stdlib unittest).

Jalankan dari akar proyek:
    .venv\\Scripts\\python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from app.classifier import PhaseClassifier
from app.config import MODELS_DIR, PHASES, SAMPLES_DIR, find_yolo_weights
from app.pipeline import (
    Instance,
    Pipeline,
    decode_image,
    masks_from_polygons,
    measure_instances,
    preprocess_image,
)
from app.recommendations import build_recommendations

SAMPLE = SAMPLES_DIR / "day-20.jpg"


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not SAMPLE.exists():
            raise unittest.SkipTest(f"Contoh tidak ditemukan: {SAMPLE}")
        cls.data = SAMPLE.read_bytes()
        cls.pipeline = Pipeline()

    def test_decode_and_preprocess(self) -> None:
        original = decode_image(self.data)
        resized, preprocessed = preprocess_image(original)
        self.assertGreater(max(resized.shape[:2]), 0)
        self.assertLessEqual(max(resized.shape[:2]), 640)
        self.assertEqual(resized.shape, preprocessed.shape)

    def test_primary_weights_is_best_pt(self) -> None:
        best = MODELS_DIR / "best.pt"
        if not best.exists():
            self.skipTest("models/best.pt belum tersedia")
        self.assertEqual(find_yolo_weights(), best)

    def test_checkpoint_metadata_is_read(self) -> None:
        if find_yolo_weights() is None:
            self.skipTest("Bobot segmentasi belum tersedia")
        info = Pipeline().yolo.model_info()
        self.assertIsNotNone(info)
        self.assertEqual(info["classes"], ["daun-pakcoy"])
        self.assertEqual(info["imgsz"], 640)
        mask_metrics = (info.get("metrics") or {}).get("mask") or {}
        self.assertIsNotNone(mask_metrics.get("map50"))

    def test_mask_polygon_keeps_original_position(self) -> None:
        polygon = np.array([[10, 20], [30, 20], [30, 40], [10, 40]])
        instances = masks_from_polygons([polygon], 100, 80, [0.9])
        self.assertEqual(len(instances), 1)
        mask = instances[0].mask
        self.assertEqual(mask.shape, (100, 80))
        ys, xs = np.nonzero(mask)
        self.assertGreaterEqual(int(xs.min()), 10)
        self.assertLessEqual(int(xs.max()), 30)
        self.assertGreaterEqual(int(ys.min()), 20)
        self.assertLessEqual(int(ys.max()), 40)
        self.assertAlmostEqual(float(instances[0].confidence or 0), 0.9)

    def test_analyze_classic_end_to_end(self) -> None:
        result = self.pipeline.analyze(
            self.data, "day-20.jpg", engine="classic"
        )
        self.assertEqual(result["engine"]["used"], "classic")
        self.assertIn("original", result["image"])
        self.assertIn("preprocessed", result["image"])
        self.assertIn("annotated", result["image"])
        self.assertGreater(len(result["leaves"]), 0)
        self.assertEqual(
            result["aggregated"]["leaf_count"], len(result["leaves"])
        )
        self.assertEqual(
            result["aggregated"]["leaf_count"],
            result["aggregated_raw"]["leaf_count"],
        )
        for leaf in result["leaves"]:
            self.assertGreater(leaf["length"], 0)
            self.assertGreater(leaf["width"], 0)
            self.assertIn("width_visible", leaf)
            self.assertIsInstance(leaf["occluded"], bool)
            self.assertGreaterEqual(leaf["width"], leaf["width_visible"])
            self.assertGreater(leaf["area"], 0)
            self.assertGreater(leaf["ratio"], 0)
            self.assertTrue(leaf["mask"].startswith("data:image/png;base64,"))
        prediction = result["prediction"]
        self.assertIn(prediction["phase"], PHASES)
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=6)
        self.assertIn("summary", result["recommendations"])
        self.assertTrue(result["recommendations"]["items"])

    def test_calibration_changes_display_units_only(self) -> None:
        raw = self.pipeline.analyze(self.data, "day-20.jpg", engine="classic")
        calibrated = self.pipeline.analyze(
            self.data, "day-20.jpg", engine="classic", pixels_per_cm=38.5
        )
        self.assertFalse(raw["units"]["calibrated"])
        self.assertTrue(calibrated["units"]["calibrated"])
        self.assertEqual(calibrated["units"]["length"], "cm")
        self.assertEqual(calibrated["units"]["area"], "cm²")
        # Fitur klasifikasi tetap identik karena dihitung dalam piksel.
        self.assertEqual(raw["aggregated_raw"], calibrated["aggregated_raw"])
        self.assertEqual(
            raw["prediction"]["phase"], calibrated["prediction"]["phase"]
        )

    def test_classifier_handles_zero_leaves(self) -> None:
        classifier = PhaseClassifier()
        features = {
            column: 0.0
            for column in (
                "leaf_count",
                "mean_length",
                "mean_width",
                "max_length",
                "max_width",
                "total_area",
                "mean_ratio",
            )
        }
        prediction = classifier.predict(features, {})
        self.assertIsNone(prediction["phase"])
        self.assertEqual(prediction["confidence"], 0.0)

    def test_recommendation_without_phase(self) -> None:
        result = build_recommendations(
            phase=None,
            features={},
            context={},
            confidence=0.0,
            is_demo=True,
            leaf_count=0,
        )
        self.assertIn("belum dapat ditentukan", result["summary"])
        self.assertEqual(result["items"][0]["type"], "warning")

    def _ellipse_mask(self, cut: bool) -> np.ndarray:
        height = width = 240
        ys, xs = np.mgrid[0:height, 0:width]
        center_x, center_y = 120, 120
        inside = ((xs - center_x) / 40.0) ** 2 + ((ys - center_y) / 80.0) ** 2 <= 1.0
        if cut:
            inside &= xs >= center_x
        return inside.astype(np.uint8)

    def test_occluded_leaf_width_is_reconstructed(self) -> None:
        mask = self._ellipse_mask(cut=True)
        leaves = measure_instances([Instance(mask=mask, confidence=0.9)])
        self.assertEqual(len(leaves), 1)
        leaf = leaves[0]
        self.assertTrue(leaf.occluded)
        self.assertAlmostEqual(leaf.length_px, 160.0, delta=4.0)
        self.assertAlmostEqual(leaf.width_visible_px, 40.0, delta=4.0)
        # Lebar bagian yang terlihat sekitar 40 px; setelah rekonstruksi
        # cermin, lebar penuh kembali mendekati 80 px.
        self.assertGreater(leaf.width_px, 68.0)
        self.assertLess(leaf.width_px, 92.0)

    def test_full_leaf_width_is_not_doubled(self) -> None:
        mask = self._ellipse_mask(cut=False)
        leaves = measure_instances([Instance(mask=mask, confidence=0.9)])
        self.assertEqual(len(leaves), 1)
        leaf = leaves[0]
        self.assertFalse(leaf.occluded)
        self.assertAlmostEqual(leaf.width_px, 80.0, delta=4.0)
        self.assertAlmostEqual(leaf.width_px, leaf.width_visible_px, delta=4.0)


if __name__ == "__main__":
    unittest.main()
