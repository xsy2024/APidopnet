from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="finn_release_test_") as tmp:
        out_dir = Path(tmp) / "toy_example"
        cmd = [
            sys.executable,
            str(ROOT / "examples" / "run_toy_gene_panel.py"),
            "--out-dir",
            str(out_dir),
        ]
        completed = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)

        summary_path = out_dir / "toy_summary.csv"
        if not summary_path.exists():
            raise SystemExit(f"Smoke test failed: missing {summary_path}")

        summary = pd.read_csv(summary_path)
        required_columns = {"method", "toy_mse"}
        if not required_columns.issubset(summary.columns):
            raise SystemExit(f"Smoke test failed: missing columns {sorted(required_columns)}")
        if len(summary) < 2:
            raise SystemExit("Smoke test failed: expected at least two methods")
        if summary["toy_mse"].isna().any() or (summary["toy_mse"] < 0).any():
            raise SystemExit("Smoke test failed: invalid toy MSE values")

    print("Smoke test OK")


if __name__ == "__main__":
    main()
