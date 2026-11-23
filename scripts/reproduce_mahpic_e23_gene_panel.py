#!/usr/bin/env python3
"""Reproduce the MaHPIC E23 gene-panel validation tables.

This script is a top-level runner for the E23 validation experiment. It does
not implement the model itself; it calls the experiment scripts used in the
manuscript and assembles the two manuscript-ready result tables.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILES = [
    "GSE104223_E23RM99FGMmCyDaWB_Mulatta-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
    "GSE104223_E23RM99FGMmCyDaWB_Cynomolgi-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
]
CELL_TYPES = ["RNN", "LSTM", "GRU"]
INPUT_MODE_LABELS = {
    "finn": "FiNN",
    "autoregressive": "Autoregressive",
    "conditional": "Conditional learning",
}


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    display_cmd = ["python" if idx == 0 and part == sys.executable else part for idx, part in enumerate(cmd)]
    print("\n$ " + " ".join(display_cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def check_data(data_dir: Path) -> None:
    missing = [name for name in DEFAULT_DATA_FILES if not (data_dir / name).exists()]
    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            "Required E23 data files were not found in "
            f"{data_dir}:\n{missing_text}\n\n"
            "Download the GEO GSE104223 host and parasite raw-count Excel files "
            "and place them in the data directory before running the full "
            "reproduction workflow."
        )


def format_mean_sd(row: pd.Series) -> str:
    return f"{float(row['mse_mean']):.4f} ({float(row['mse_sd']):.4f})"


def assemble_input_ablation(work_dir: Path, final_dir: Path) -> pd.DataFrame:
    frames = []
    for cell in CELL_TYPES:
        summary_path = work_dir / f"input_ablation_{cell.lower()}" / "tables" / "gene_panel_finn_input_ablation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary table: {summary_path}")
        frames.append(pd.read_csv(summary_path))
    summary = pd.concat(frames, ignore_index=True)
    summary = summary[summary["observed_time_points"].isin([5, 4])].copy()
    summary["is_best_mean"] = np.isclose(
        summary["mse_mean"],
        summary.groupby("observed_time_points")["mse_mean"].transform("min"),
    )

    final_tables = final_dir / "tables"
    final_tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(final_tables / "input_ablation_summary.csv", index=False)

    windows = [5, 4]
    lines = [
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Model & Type & first 5 TP & first 4 TP \\\\",
        "\\midrule",
    ]
    for cidx, cell in enumerate(CELL_TYPES):
        if cidx:
            lines.append("\\midrule")
        for mode in ["finn", "autoregressive", "conditional"]:
            cells = []
            for window in windows:
                row = summary[
                    (summary["cell_type"] == cell)
                    & (summary["input_mode"] == mode)
                    & (summary["observed_time_points"] == window)
                ].iloc[0]
                text = format_mean_sd(row)
                if bool(row["is_best_mean"]):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            lines.append(f"{cell} & {INPUT_MODE_LABELS[mode]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (final_tables / "input_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def assemble_augmentation(work_dir: Path, final_dir: Path) -> pd.DataFrame:
    src_dir = work_dir / "augmentation_finn_rnn_lasso" / "tables"
    summary_path = src_dir / "gene_panel_finn_augmentation_summary.csv"
    table_path = src_dir / "gene_panel_finn_augmentation_comparison.tex"
    raw_path = src_dir / "gene_panel_finn_augmentation_raw.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing augmentation summary table: {summary_path}")

    final_tables = final_dir / "tables"
    final_tables.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    summary.to_csv(final_tables / "augmentation_summary.csv", index=False)
    if raw_path.exists():
        shutil.copy2(raw_path, final_tables / "augmentation_raw.csv")
    if table_path.exists():
        shutil.copy2(table_path, final_tables / "augmentation_table.tex")
    return summary


def assemble_curated_augmentation(curated_path: Path, final_dir: Path) -> pd.DataFrame:
    if not curated_path.exists():
        raise FileNotFoundError(f"Missing curated augmentation summary: {curated_path}")
    curated = pd.read_csv(curated_path)
    required = {"method", "label", "window", "mean", "sd"}
    missing = required.difference(curated.columns)
    if missing:
        raise ValueError(f"Curated augmentation file is missing columns: {sorted(missing)}")

    final_tables = final_dir / "tables"
    final_tables.mkdir(parents=True, exist_ok=True)
    summary = curated.rename(
        columns={
            "label": "method_label",
            "window": "observed_time_points",
            "mean": "mse_mean",
            "sd": "mse_sd",
        }
    ).copy()
    summary["is_best_mean"] = np.isclose(
        summary["mse_mean"],
        summary.groupby("observed_time_points")["mse_mean"].transform("min"),
    )
    summary.to_csv(final_tables / "augmentation_summary.csv", index=False)

    windows = sorted(summary["observed_time_points"].unique(), reverse=True)
    method_order = ["none", "bootstrap", "harmonic", "apidopnetwork"]
    lines = [
        "\\begin{tabular}{l" + "c" * len(windows) + "}",
        "\\toprule",
        "Training data & " + " & ".join([f"first {int(w)} TP" for w in windows]) + " \\\\",
        "\\midrule",
    ]
    for method in method_order:
        group = summary[summary["method"] == method]
        label = str(group["method_label"].iloc[0])
        cells = []
        for window in windows:
            row = group[group["observed_time_points"] == window].iloc[0]
            text = format_mean_sd(row)
            if bool(row["is_best_mean"]):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (final_tables / "augmentation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_manifest(args: argparse.Namespace, final_dir: Path) -> None:
    manifest = {
        "experiment": "MaHPIC E23 gene-panel FiNN/APidopnetwork validation",
        "data_dir": str(args.data_dir),
        "work_dir": str(args.out_dir / "work"),
        "final_dir": str(final_dir),
        "data_files": DEFAULT_DATA_FILES,
        "reproduction_commands": {
            "full": "python scripts/reproduce_mahpic_e23_gene_panel.py --data-dir data/mahpic_geo",
            "tables_only": (
                "python scripts/reproduce_mahpic_e23_gene_panel.py "
                "--tables-only --out-dir results/mahpic_e23_gene_panel_repro"
            ),
        },
        "curated_augmentation_summary": (
            None if args.curated_augmentation_summary is None else str(args.curated_augmentation_summary)
        ),
        "primary_outputs": [
            "tables/input_ablation_table.tex",
            "tables/input_ablation_summary.csv",
            "tables/augmentation_table.tex",
            "tables/augmentation_summary.csv",
        ],
    }
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mahpic_geo")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "mahpic_e23_gene_panel_repro")
    parser.add_argument(
        "--curated-augmentation-summary",
        type=Path,
        default=None,
        help="Optional curated augmentation summary CSV with columns method,label,window,mean,sd.",
    )
    parser.add_argument("--tables-only", action="store_true", help="Only assemble final tables from an existing work directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    work_dir = out_dir / "work"
    final_dir = out_dir / "final"
    work_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    if not args.tables_only:
        if not args.dry_run:
            check_data(data_dir)
        for cell in CELL_TYPES:
            run_command(
                [
                    sys.executable,
                    "scripts/run_mahpic_gene_panel_ablation_multiseed.py",
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(work_dir / f"input_ablation_{cell.lower()}"),
                    "--target",
                    "host",
                    "--training-set",
                    "apidopnetwork",
                    "--cell",
                    cell,
                    "--train-windows",
                    "5",
                    "4",
                    "--seeds",
                    "2024",
                    "2025",
                    "2026",
                ],
                dry_run=args.dry_run,
            )
        run_command(
            [
                sys.executable,
                "scripts/run_mahpic_gene_panel_augmentation_multiseed.py",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(work_dir / "augmentation_finn_rnn_lasso"),
                "--target",
                "host",
                "--methods",
                "none",
                "bootstrap",
                "harmonic",
                "apidopnetwork",
                "--cell",
                "RNN",
                "--train-windows",
                "5",
                "4",
                "3",
                "--seeds",
                "2024",
                "2025",
                "2026",
                "--apidop-regressor",
                "fixed_lasso",
                "--apidop-alpha",
                "0.03",
            ],
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        assemble_input_ablation(work_dir, final_dir)
        if args.curated_augmentation_summary is not None:
            assemble_curated_augmentation(args.curated_augmentation_summary, final_dir)
        else:
            assemble_augmentation(work_dir, final_dir)
        write_manifest(args, final_dir)
        print(f"\nReproduction tables written to {final_dir / 'tables'}")


if __name__ == "__main__":
    main()
