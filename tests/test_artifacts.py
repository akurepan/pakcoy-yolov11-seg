"""Uji integritas artefak yang disertakan di repositori.

Uji ini menjaga dua hal yang pernah bermasalah:
1. `hasil/pakcoy_artifacts.zip` harus bisa diekstrak (CRC valid).
2. Bobot YOLO di dalam zip harus identik dengan `models/best.pt` yang dipakai
   aplikasi, supaya tidak ada dua versi model yang berbeda.
"""

from __future__ import annotations

import hashlib
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "hasil" / "pakcoy_artifacts.zip"
WEIGHTS_PATH = ROOT / "models" / "best.pt"


class ArtifactIntegrityTest(unittest.TestCase):
    def test_zip_exists_and_is_readable(self) -> None:
        self.assertTrue(ZIP_PATH.is_file(), f"tidak ada: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH) as archive:
            self.assertIsNone(
                archive.testzip(),
                "salah satu entri zip rusak (CRC tidak cocok)",
            )
            self.assertLessEqual(
                {"best.pt", "last.pt", "data.yaml"},
                set(archive.namelist()),
            )

    def test_zip_best_matches_application_weights(self) -> None:
        self.assertTrue(WEIGHTS_PATH.is_file(), f"tidak ada: {WEIGHTS_PATH}")
        expected = hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest()
        with zipfile.ZipFile(ZIP_PATH) as archive:
            actual = hashlib.sha256(archive.read("best.pt")).hexdigest()
        self.assertEqual(expected, actual)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
