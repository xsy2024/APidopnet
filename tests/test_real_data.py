#!/usr/bin/env python3
"""Exercise real-data CLI paths with reduced settings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_data import verify_data


def run(script: str, arguments: list[str]) -> None:
    print(f"Checking {Path(script).name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True)


def check_summary(path: Path, expected_rows: int) -> None:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"Unexpected summary row count: {path.name}")
    values = frame[["mse_mean", "mse_sd"]].to_numpy()
    if not np.isfinite(values).all() or (frame["mse_mean"] < 0).any():
        raise ValueError(f"Invalid summary metrics: {path.name}")
    # A single future point in one held-out animal has zero R2 denominator.
    one_step = frame["observed_time_points"] == 5
    if not frame.loc[one_step, ["r2_mean", "r2_sd"]].isna().all().all():
        raise ValueError(f"Expected undefined one-step R2: {path.name}")
    if not np.isfinite(frame.loc[~one_step, ["r2_mean", "r2_sd"]].to_numpy()).all():
        raise ValueError(f"Invalid multi-step R2: {path.name}")
    if (frame["n_runs"] < 2).any():
        raise ValueError(f"Insufficient animal folds: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mahpic_geo")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    verify_data(data_dir, ROOT / "data" / "manifest.json")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="finn_real_smoke_") as tmp:
        scratch = Path(tmp)
        os.environ.setdefault("MPLCONFIGDIR", str(scratch / "mplconfig"))
        work = scratch / "work"
        common = ["--data-dir", str(data_dir), "--target", "host", "--seeds", "2024",
                  "--n-panel-genes", "4", "--n-generated", "4", "--hidden-dim", "4",
                  "--epochs", "1", "--batch-size", "4"]
        for cell in ("RNN", "LSTM", "GRU"):
            run("scripts/run_mahpic_gene_panel_ablation_multiseed.py", [
                *common, "--out-dir", str(work / f"input_ablation_{cell.lower()}"),
                "--cell", cell, "--train-windows", "5", "4",
            ])
        run("scripts/run_mahpic_gene_panel_augmentation_multiseed.py", [
            *common, "--out-dir", str(work / "augmentation_finn_rnn_lasso"),
            "--cell", "RNN", "--train-windows", "5", "4", "3",
            "--apidop-regressor", "fixed_lasso", "--apidop-alpha", "0.03",
        ])
        run("scripts/reproduce_mahpic_e23_gene_panel.py", [
            "--data-dir", str(data_dir), "--out-dir", str(scratch), "--tables-only",
        ])
        tables = scratch / "final" / "tables"
        check_summary(tables / "input_ablation_summary.csv", 18)
        check_summary(tables / "augmentation_summary.csv", 12)
        for name in ("input_ablation_table.tex", "augmentation_table.tex"):
            content = (tables / name).read_text(encoding="utf-8")
            if "\\begin{tabular}" not in content or "\\end{tabular}" not in content:
                raise ValueError(f"Invalid LaTeX table: {name}")

        from run_mahpic_regeneration_prediction import load_dataset

        cache = scratch / "external_cache"
        cache.mkdir()
        external = load_dataset("E25", data_dir, cache)
        if external.meta.empty or external.meta["animal"].nunique() < 2:
            raise ValueError("Missing E25 sample metadata")
        for expression in (external.host, external.parasite):
            if expression.empty or not np.isfinite(expression.to_numpy()).all():
                raise ValueError("Invalid E25 expression data")
        print(f"E25 loader OK: {len(external.meta)} samples")
    print("Real-data smoke test OK")


if __name__ == "__main__":
    main()
