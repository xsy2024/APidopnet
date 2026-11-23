#!/usr/bin/env python3
"""Check that the submission release is clean and reviewer-ready."""

from __future__ import annotations

import sys
from pathlib import Path
import zipfile

from verify_data import verify_data


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TEXT_PATTERNS = [
    "/" + "Users/",
    "Down" + "loads",
    "Co" + "dex",
    "Library/" + "Containers",
    "Temporary" + "Items",
    "We" + "Chat",
    "we" + "chat",
    "微" + "信",
]

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".env",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}


def iter_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]


def fail(message: str) -> None:
    print(f"Release check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_forbidden_paths() -> None:
    hits: list[str] = []
    for path in iter_files():
        if path.relative_to(ROOT) == Path("scripts/check_release.py"):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT)
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern in text:
                hits.append(f"{rel}: contains {pattern!r}")
    if hits:
        fail("local/private path patterns found:\n" + "\n".join(hits))


def check_generated_artifacts() -> None:
    generated = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir() and path.name in {"__pycache__", ".matplotlib"}:
            generated.append(str(path.relative_to(ROOT)))
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            generated.append(str(path.relative_to(ROOT)))
    if generated:
        fail("generated cache artifacts found:\n" + "\n".join(generated))


def check_data_directory() -> None:
    try:
        count, total = verify_data(ROOT / "data" / "mahpic_geo", ROOT / "data" / "manifest.json")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        fail(str(exc))
    print(f"Data integrity OK: {count} workbooks, {total:,} bytes")


def check_results_directory() -> None:
    results_dir = ROOT / "results"
    allowed = {".gitkeep"}
    unexpected = [
        path.relative_to(ROOT)
        for path in results_dir.rglob("*")
        if path.is_file() and path.name not in allowed
    ]
    if unexpected:
        fail(
            "generated result files are present under results/; "
            "the release should ship code, not local outputs:\n"
            + "\n".join(str(path) for path in unexpected)
        )


def main() -> None:
    check_forbidden_paths()
    check_generated_artifacts()
    check_data_directory()
    check_results_directory()
    print("Release hygiene OK")


if __name__ == "__main__":
    main()
