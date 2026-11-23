#!/usr/bin/env python3
"""E23 FiNN input-mechanism ablation with APidopnetwork-regenerated training data.

This experiment matches the manuscript-style comparison:

    FiNN: autoregressive target input + conditional input
    Autoregressive: autoregressive target input only
    Conditional learning: conditional input only

APidopnetwork is used only to regenerate paired host--parasite training
trajectories. It is not used as an extra recurrent-network feature.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "finn_apidopnetwork_mplconfig"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_mahpic_regeneration_prediction import (
    DEFAULT_TIME_GRID,
    ModuleSequences,
    Standardizer,
    TorchSequenceModel,
    fit_module_sequences,
    load_dataset,
    make_regenerated_sequences,
    subset_dataset,
)


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir


def time_features(time: np.ndarray, n_seq: int) -> np.ndarray:
    t_scaled = (time / max(float(time.max()), 1.0))[None, :, None]
    omega = 2.0 * np.pi / 48.0
    sin_t = np.sin(omega * time)[None, :, None]
    cos_t = np.cos(omega * time)[None, :, None]
    return np.tile(np.concatenate([t_scaled, sin_t, cos_t], axis=2), (n_seq, 1, 1))


def split_target_condition(mod: ModuleSequences, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "parasite":
        return mod.parasite, mod.host
    if target == "host":
        return mod.host, mod.parasite
    raise ValueError(target)


def build_input(
    target_prev: np.ndarray,
    condition_next: np.ndarray,
    time_next: np.ndarray,
    mode: str,
) -> np.ndarray:
    features = [time_features(time_next, target_prev.shape[0])]
    if mode in {"finn", "autoregressive"}:
        features.insert(0, target_prev)
    if mode in {"finn", "conditional"}:
        insert_at = 1 if mode == "finn" else 0
        features.insert(insert_at, condition_next)
    return np.concatenate(features, axis=2)


def make_transition_data(
    target_seq: np.ndarray,
    condition_seq: np.ndarray,
    time: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = build_input(target_seq[:, :-1, :], condition_seq[:, 1:, :], time[1:], mode)
    targets = target_seq[:, 1:, :]
    return inputs, targets


def fit_model(
    target_train: np.ndarray,
    condition_train: np.ndarray,
    time: np.ndarray,
    mode: str,
    cell_type: str,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[TorchSequenceModel, Standardizer, Standardizer]:
    torch.manual_seed(seed)
    x_raw, y_raw = make_transition_data(target_train, condition_train, time, mode)
    x_scaler = Standardizer.fit(x_raw)
    y_scaler = Standardizer.fit(y_raw)
    x = torch.tensor(x_scaler.transform(x_raw).astype(np.float32))
    y = torch.tensor(y_scaler.transform(y_raw).astype(np.float32))
    model = TorchSequenceModel(x.shape[-1], hidden_dim, y.shape[-1], cell_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed + 17)
    n_seq = x.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n_seq)
        for start in range(0, n_seq, batch_size):
            idx = order[start : start + batch_size]
            pred, _ = model(x[idx])
            loss = loss_fn(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
    return model, x_scaler, y_scaler


def forecast(
    model: TorchSequenceModel,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    target_seq: np.ndarray,
    condition_seq: np.ndarray,
    time: np.ndarray,
    mode: str,
    train_window: int,
) -> np.ndarray:
    model.eval()
    n_seq, n_time, _ = target_seq.shape
    pred_seq = target_seq.copy()
    hidden: object | None = None
    current = target_seq[:, 0, :]
    with torch.no_grad():
        for k in range(train_window - 1):
            inp = build_input(current[:, None, :], condition_seq[:, k + 1 : k + 2, :], time[k + 1 : k + 2], mode)
            inp_scaled = torch.tensor(x_scaler.transform(inp).astype(np.float32))
            _, hidden = model(inp_scaled, hidden)
            current = target_seq[:, k + 1, :]

        current = target_seq[:, train_window - 1, :]
        for k in range(train_window - 1, n_time - 1):
            inp = build_input(current[:, None, :], condition_seq[:, k + 1 : k + 2, :], time[k + 1 : k + 2], mode)
            inp_scaled = torch.tensor(x_scaler.transform(inp).astype(np.float32))
            pred_scaled, hidden = model(inp_scaled, hidden)
            pred = y_scaler.inverse(pred_scaled.numpy())[:, 0, :]
            pred_seq[:, k + 1, :] = pred
            current = pred
    return pred_seq


def mse_r2(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((true - pred) ** 2))
    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - true.mean(axis=(0, 1), keepdims=True)) ** 2))
    if sst < 1e-8:
        return mse, float("nan")
    return mse, 1.0 - sse / sst


def make_training_pair(
    train_mod: ModuleSequences,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    time = train_mod.time_grid
    if args.training_set == "original":
        return split_target_condition(train_mod, args.target)

    regenerated = make_regenerated_sequences(
        np.random.default_rng(args.seed + 1201),
        train_mod.host,
        train_mod.parasite,
        time,
        "ap",
        args.n_generated,
    )
    host_dim = train_mod.host.shape[-1]
    host_aug = np.concatenate([train_mod.host, regenerated[:, :, :host_dim]], axis=0)
    parasite_aug = np.concatenate([train_mod.parasite, regenerated[:, :, host_dim:]], axis=0)
    aug_mod = ModuleSequences(train_mod.experiment, train_mod.animals, time, host_aug, parasite_aug)
    return split_target_condition(aug_mod, args.target)


def evaluate_fold(
    train_mod: ModuleSequences,
    test_mod: ModuleSequences,
    fold: str,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    target_train, condition_train = make_training_pair(train_mod, args)
    target_test, condition_test = split_target_condition(test_mod, args.target)
    rows: list[dict[str, float | int | str]] = []
    for train_window in args.train_windows:
        future_slice = slice(train_window, len(train_mod.time_grid))
        for mode in ["finn", "autoregressive", "conditional"]:
            for cell_type in args.cells:
                model, x_scaler, y_scaler = fit_model(
                    target_train,
                    condition_train,
                    train_mod.time_grid,
                    mode,
                    cell_type,
                    args.seed + 1000 + len(fold) + train_window + 13 * len(mode) + 37 * len(cell_type),
                    args.hidden_dim,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                )
                pred = forecast(model, x_scaler, y_scaler, target_test, condition_test, train_mod.time_grid, mode, train_window)
                mse, r2 = mse_r2(target_test[:, future_slice, :], pred[:, future_slice, :])
                rows.append(
                    {
                        "fold": fold,
                        "target": args.target,
                        "training_set": args.training_set,
                        "input_mode": mode,
                        "cell_type": cell_type,
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
    return (
        raw.groupby(["target", "training_set", "input_mode", "cell_type", "observed_time_points"], sort=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_sd=("mse", "std"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )


def write_tables(summary: pd.DataFrame, table_dir: Path) -> None:
    label = {
        "finn": "FiNN",
        "autoregressive": "Autoregressive",
        "conditional": "Conditional learning",
    }
    for (target, training_set), frame in summary.groupby(["target", "training_set"], sort=False):
        windows = sorted(frame["observed_time_points"].unique(), reverse=True)
        lines = [
            "\\begin{tabular}{ll" + "c" * len(windows) + "}",
            "\\toprule",
            "Type & Model & " + " & ".join([f"first {int(w)} TP" for w in windows]) + " \\\\",
            "\\midrule",
        ]
        for mode in ["finn", "autoregressive", "conditional"]:
            for cell_type in ["RNN", "LSTM", "GRU"]:
                subset = frame[(frame["input_mode"] == mode) & (frame["cell_type"] == cell_type)]
                cells = []
                for window in windows:
                    row = subset[subset["observed_time_points"] == window].iloc[0]
                    cells.append(f"{float(row['mse_mean']):.4f}")
                lines.append(f"{label[mode]} & {cell_type} & " + " & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        path = table_dir / f"mahpic_input_ablation_{target}_{training_set}.tex"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        detail = [
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Type & Model & " + " & ".join([f"first {int(w)} TP" for w in windows]) + " \\\\",
            "\\midrule",
        ]
        for mode in ["finn", "autoregressive", "conditional"]:
            for cell_type in ["RNN", "LSTM", "GRU"]:
                subset = frame[(frame["input_mode"] == mode) & (frame["cell_type"] == cell_type)]
                cells = []
                for window in windows:
                    row = subset[subset["observed_time_points"] == window].iloc[0]
                    cells.append(f"{float(row['mse_mean']):.4f} ({float(row['mse_sd']):.4f})")
                detail.append(f"{label[mode]} & {cell_type} & " + " & ".join(cells) + " \\\\")
        detail.extend(["\\bottomrule", "\\end{tabular}"])
        (table_dir / f"mahpic_input_ablation_{target}_{training_set}_mean_sd.tex").write_text(
            "\n".join(detail) + "\n", encoding="utf-8"
        )


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    label = {"finn": "FiNN", "autoregressive": "Autoregressive", "conditional": "Conditional"}
    colors = {"finn": "#D95F02", "autoregressive": "#1B9E77", "conditional": "#7570B3"}
    for (target, training_set, cell_type), frame in summary.groupby(["target", "training_set", "cell_type"], sort=False):
        fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        for mode, mf in frame.groupby("input_mode", sort=False):
            mf = mf.sort_values("observed_time_points", ascending=False)
            ax.errorbar(
                mf["observed_time_points"],
                mf["mse_mean"],
                yerr=mf["mse_sd"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                color=colors[mode],
                label=label[mode],
            )
        ax.invert_xaxis()
        ax.set_xlabel("Observed prefix length (time points)")
        ax.set_ylabel("Future target MSE")
        ax.set_title(f"E23 LOO, target={target}, {training_set}, {cell_type}")
        ax.grid(True, color="#E6E6E6")
        ax.legend(frameon=False)
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{target}_{training_set}_{cell_type}")
        fig.savefig(fig_dir / f"mahpic_input_ablation_{safe}.png", dpi=240)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mahpic_geo"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mahpic_finn_input_ablation_e23"))
    parser.add_argument("--target", choices=["parasite", "host"], default="parasite")
    parser.add_argument("--training-set", choices=["apidopnetwork", "original"], default="apidopnetwork")
    parser.add_argument("--n-host-components", type=int, default=5)
    parser.add_argument("--n-parasite-components", type=int, default=5)
    parser.add_argument("--n-top-genes", type=int, default=500)
    parser.add_argument("--n-generated", type=int, default=160)
    parser.add_argument("--train-windows", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--cells", nargs="+", default=["RNN", "LSTM", "GRU"], choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--hidden-dim", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-grid", type=float, nargs="+", default=list(DEFAULT_TIME_GRID))
    args = parser.parse_args()

    table_dir, fig_dir = ensure_dirs(args.out_dir)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    e23 = load_dataset("E23", args.data_dir, cache_dir)
    rows: list[dict[str, float | int | str]] = []
    time_grid = np.asarray(args.time_grid, dtype=float)
    for animal in sorted(e23.meta["animal"].unique()):
        train = subset_dataset(e23, e23.meta["animal"] != animal)
        test = subset_dataset(e23, e23.meta["animal"] == animal)
        train_mod, test_mod = fit_module_sequences(
            train,
            test,
            args.n_host_components,
            args.n_parasite_components,
            args.n_top_genes,
            time_grid,
        )
        rows.extend(evaluate_fold(train_mod, test_mod, str(animal), args))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(table_dir / "mahpic_input_ablation_raw.csv", index=False)
    summary.to_csv(table_dir / "mahpic_input_ablation_summary.csv", index=False)
    write_tables(summary, table_dir)
    plot_summary(summary, fig_dir)
    metadata = {
        "target": args.target,
        "training_set": args.training_set,
        "time_grid": [float(x) for x in time_grid],
        "n_generated": args.n_generated,
        "note": "FiNN uses target autoregression plus conditional input; APidopnetwork is used only for data regeneration.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
