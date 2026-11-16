"""Uji tautan relatif di dokumentasi.

Mencegah tautan rusak setelah dokumen dipindahkan ke folder `docs/`.
Tautan eksternal (http/https/mailto) dan anchor saja dilewati.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    ROOT / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
    ROOT / "models" / "README.md",
    ROOT / "hasil" / "README.md",
]
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


class DocumentationLinkTest(unittest.TestCase):
    def test_relative_links_resolve(self) -> None:
        missing: list[str] = []
        for document in DOC_FILES:
            self.assertTrue(document.is_file(), f"dokumen hilang: {document}")
            text = document.read_text(encoding="utf-8")
            for raw_target in LINK_PATTERN.findall(text):
                target = raw_target.strip().split()[0].strip("<>")
                if not target or target.startswith(
                    ("http://", "https://", "mailto:", "#")
                ):
                    continue
                path_part = target.split("#", 1)[0]
                if not path_part:
                    continue
                resolved = (document.parent / path_part).resolve()
                if not resolved.exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [], "Tautan dokumentasi rusak:\n" + "\n".join(missing))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
