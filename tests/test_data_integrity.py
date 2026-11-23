from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_data import verify_data


class DataIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="finn_integrity_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.book = self.data / "sample.xlsx"
        with zipfile.ZipFile(self.book, "w") as archive:
            archive.writestr("xl/workbook.xml", "<workbook />")
        self.record = {
            "filename": self.book.name,
            "bytes": self.book.stat().st_size,
            "sha256": hashlib.sha256(self.book.read_bytes()).hexdigest(),
        }
        self.manifest = self.root / "manifest.json"
        self.write_manifest([self.record])

    def write_manifest(self, records: list[dict]) -> None:
        self.manifest.write_text(json.dumps({"schema_version": 1, "files": records}), encoding="utf-8")

    def test_valid(self) -> None:
        self.assertEqual(verify_data(self.data, self.manifest), (1, self.record["bytes"]))

    def test_missing(self) -> None:
        self.book.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_data(self.data, self.manifest)

    def test_changed_size(self) -> None:
        self.book.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Byte-size mismatch"):
            verify_data(self.data, self.manifest)

    def test_same_size_corruption(self) -> None:
        content = bytearray(self.book.read_bytes())
        content[-1] ^= 1
        self.book.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verify_data(self.data, self.manifest)

    def test_unlisted(self) -> None:
        (self.data / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            verify_data(self.data, self.manifest)

    def test_unsafe_filename(self) -> None:
        self.write_manifest([{**self.record, "filename": "../sample.xlsx"}])
        with self.assertRaisesRegex(ValueError, "basenames"):
            verify_data(self.data, self.manifest)

    def test_duplicate(self) -> None:
        self.write_manifest([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_data(self.data, self.manifest)


if __name__ == "__main__":
    unittest.main()
