#!/usr/bin/env python3
"""Verify the bundled GEO spreadsheets against the local-source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def verify_data(data_dir: Path, manifest_path: Path) -> tuple[int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported data manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Data manifest must list at least one file")
    if not data_dir.is_dir():
        raise ValueError("Data directory is missing")

    names: set[str] = set()
    total = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid manifest record")
        name = record.get("filename")
        if (not isinstance(name, str) or Path(name).name != name
                or "\\" in name or not name.endswith(".xlsx")):
            raise ValueError("Manifest filenames must be plain .xlsx basenames")
        if name in names:
            raise ValueError(f"Duplicate manifest entry: {name}")
        names.add(name)
        size, digest = record.get("bytes"), record.get("sha256")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid byte size: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256: {name}")

        path = data_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular data file: {name}")
        if path.stat().st_size != size:
            raise ValueError(f"Byte-size mismatch: {name}")
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            actual = hasher.hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt workbook archive: {name}")
            if "xl/workbook.xml" not in archive.namelist():
                raise ValueError(f"Missing workbook XML: {name}")
        total += size

    extra = [str(path.relative_to(data_dir)) for path in data_dir.rglob("*")
             if path.is_file() and path.relative_to(data_dir).as_posix()
             not in names | {".gitkeep", "README.md"}]
    if extra:
        raise ValueError("Unlisted data files: " + ", ".join(sorted(extra)))
    return len(records), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mahpic_geo")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.json")
    args = parser.parse_args()
    try:
        count, total = verify_data(args.data_dir, args.manifest)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Data verification failed: {exc}\n")
    print(f"Data integrity OK: {count} workbooks, {total:,} bytes")


if __name__ == "__main__":
    main()
