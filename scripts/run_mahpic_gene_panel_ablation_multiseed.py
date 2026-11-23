#!/usr/bin/env python3
"""Multi-seed FiNN input ablation on the E23 raw gene panel.

This script isolates the recurrent input mechanism:

    FiNN:               H_{k-1}, P_k, time_k -> H_k
    Autoregressive:     H_{k-1}, time_k      -> H_k
    Conditional:        P_k, time_k          -> H_k

The training set can be the original observed trajectories or the
APidopnetwork-regenerated trajectories.  The default follows the main
pipeline: raw 20-gene panel + APidopnetwork-regenerated paired trajectories +
RNN cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_mahpic_finn_input_ablation import (
    fit_model,
    forecast,
    mse_r2,
    split_target_condition,
)
from run_mahpic_gene_panel_input_ablation import (
    fit_gene_panel_sequences,
    make_training_pair_panel,
    select_gene_panel,
)
from run_mahpic_regeneration_prediction import DEFAULT_TIME_GRID, load_dataset, subset_dataset


MODE_LABELS = {
    "finn": "FiNN",
    "autoregressive": "Autoregressive",
    "conditional": "Conditional learning",
}


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir


def evaluate_seed_fold(train_mod, test_mod, fold: str, seed: int, args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    args.seed = seed
    target_train, condition_train = make_training_pair_panel(train_mod, args)
    target_test, condition_test = split_target_condition(test_mod, args.target)
    rows: list[dict[str, float | int | str]] = []
    for train_window in args.train_windows:
        future_slice = slice(train_window, len(train_mod.time_grid))
        for mode in ["finn", "autoregressive", "conditional"]:
            model, x_scaler, y_scaler = fit_model(
                target_train,
                condition_train,
                train_mod.time_grid,
                mode,
                args.cell,
                seed + 1000 + len(fold) + train_window + 13 * len(mode),
                args.hidden_dim,
                args.epochs,
                args.batch_size,
                args.lr,
            )
            pred = forecast(model, x_scaler, y_scaler, target_test, condition_test, train_mod.time_grid, mode, train_window)
            mse, r2 = mse_r2(target_test[:, future_slice, :], pred[:, future_slice, :])
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "target": args.target,
                    "training_set": args.training_set,
                    "input_mode": mode,
                    "cell_type": args.cell,
                    "observed_time_points": int(train_window),
                    "future_time_points": int(len(train_mod.time_grid) - train_window),
                    "mse": mse,
                    "r2": r2,
                    "n_original_train_trajectories": int(train_mod.host.shape[0]),
                    "n_generated_trajectories": int(0 if args.training_set == "original" else args.n_generated),
                }
            )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["target", "training_set", "input_mode", "cell_type", "observed_time_points"], sort=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_sd=("mse", "std"),
            mse_median=("mse", "median"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            n_runs=("mse", "count"),
        )
        .reset_index()
    )
    best = summary.groupby("observed_time_points")["mse_mean"].transform("min")
    summary["is_best_mean"] = np.isclose(summary["mse_mean"], best)
    return summary


def write_summary_tables(summary: pd.DataFrame, raw: pd.DataFrame, table_dir: Path) -> None:
    windows = sorted(summary["observed_time_points"].unique(), reverse=True)
    lines = [
        "\\begin{tabular}{l" + "c" * len(windows) + "}",
        "\\toprule",
        "Type & " + " & ".join([f"first {int(w)} TP" for w in windows]) + " \\\\",
        "\\midrule",
    ]
    for mode in ["finn", "autoregressive", "conditional"]:
        cells = []
        for window in windows:
            row = summary[(summary["input_mode"] == mode) & (summary["observed_time_points"] == window)].iloc[0]
            text = f"{float(row['mse_mean']):.4f} ({float(row['mse_sd']):.4f})"
            if bool(row["is_best_mean"]):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{MODE_LABELS[mode]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / "gene_panel_finn_input_ablation_multiseed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    win_table = (
        raw.groupby(["observed_time_points", "seed", "fold"], sort=False)
        .apply(lambda g: g.loc[g["mse"].idxmin(), "input_mode"], include_groups=False)
        .reset_index(name="winner")
        .groupby(["observed_time_points", "winner"])
        .size()
        .reset_index(name="wins")
    )
    win_table.to_csv(table_dir / "gene_panel_finn_input_ablation_wins.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mahpic_geo"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mahpic_gene_panel_ablation_multiseed_e23_host_target"))
    parser.add_argument("--target", choices=["host", "parasite"], default="host")
    parser.add_argument("--training-set", choices=["apidopnetwork", "original"], default="apidopnetwork")
    parser.add_argument("--n-panel-genes", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--n-generated", type=int, default=160)
    parser.add_argument("--host-gaussian-shrinkage", type=float, default=0.35)
    parser.add_argument("--train-windows", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--cell", choices=["RNN", "LSTM", "GRU"], default="RNN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--hidden-dim", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--time-grid", type=float, nargs="+", default=list(DEFAULT_TIME_GRID))
    args = parser.parse_args()

    table_dir, _ = ensure_dirs(args.out_dir)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    e23 = load_dataset("E23", args.data_dir, cache_dir)
    time_grid = np.asarray(args.time_grid, dtype=float)
    host_genes, host_stats = select_gene_panel(e23.host, args.n_panel_genes, args.top_fraction)
    parasite_genes, parasite_stats = select_gene_panel(e23.parasite, args.n_panel_genes, args.top_fraction)
    host_stats.to_csv(table_dir / "selected_host_gene_panel.csv", index=False)
    parasite_stats.to_csv(table_dir / "selected_parasite_gene_panel.csv", index=False)

    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        for animal in sorted(e23.meta["animal"].unique()):
            train = subset_dataset(e23, e23.meta["animal"] != animal)
            test = subset_dataset(e23, e23.meta["animal"] == animal)
            train_mod, test_mod = fit_gene_panel_sequences(train, test, host_genes, parasite_genes, time_grid)
            rows.extend(evaluate_seed_fold(train_mod, test_mod, str(animal), seed, args))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(table_dir / "gene_panel_finn_input_ablation_raw.csv", index=False)
    summary.to_csv(table_dir / "gene_panel_finn_input_ablation_summary.csv", index=False)
    write_summary_tables(summary, raw, table_dir)
    metadata = {
        "target": args.target,
        "training_set": args.training_set,
        "cell": args.cell,
        "seeds": args.seeds,
        "feature_mode": "raw_gene_panel",
        "n_panel_genes_per_species": args.n_panel_genes,
        "n_generated": args.n_generated,
        "input_modes": {
            "finn": "H_{k-1}, P_k, time_k -> H_k",
            "autoregressive": "H_{k-1}, time_k -> H_k",
            "conditional": "P_k, time_k -> H_k",
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
