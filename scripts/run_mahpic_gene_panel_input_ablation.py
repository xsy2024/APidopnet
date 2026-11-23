#!/usr/bin/env python3
"""E23 gene-panel FiNN input ablation.

This is a companion to ``run_mahpic_finn_input_ablation.py``.  Instead of
using high-variance genes followed by PCA module construction, it fixes an
interpretable gene panel selected from all E23 animals:

    host genes: top expression/variance candidates, n ~= 20
    parasite genes: top expression/variance candidates, n ~= 20

The selected raw gene trajectories are then used directly as the state and
observation coordinates in the APidopnetwork-regeneration + FiNN ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from run_mahpic_finn_input_ablation import (
    ensure_dirs,
    fit_model,
    forecast,
    mse_r2,
    plot_summary,
    split_target_condition,
    summarize,
    write_tables,
)
from run_mahpic_regeneration_prediction import (
    DEFAULT_TIME_GRID,
    ExpressionDataset,
    ModuleSequences,
    Standardizer,
    interpolate_by_animal,
    load_dataset,
    subset_dataset,
)


OMEGA = 2.0 * np.pi / 48.0
RIDGE_ALPHAS = np.logspace(-4, 3, 30)
ELASTIC_ALPHAS = np.logspace(-3, 1, 24)
LASSO_ALPHAS = np.logspace(-3, 1, 24)


def select_gene_panel(
    expr: pd.DataFrame,
    n_genes: int,
    top_fraction: float,
) -> tuple[list[str], pd.DataFrame]:
    """Select a compact panel with high expression and/or high variance.

    A selected gene is required to be in the top ``top_fraction`` of either
    mean expression or variance.  Candidates satisfying both criteria are
    prioritized, followed by genes with the largest combined percentile score.
    """
    mean_expr = expr.mean(axis=0)
    var_expr = expr.var(axis=0)
    mean_pct = mean_expr.rank(pct=True)
    var_pct = var_expr.rank(pct=True)
    cutoff = 1.0 - top_fraction
    stats = pd.DataFrame(
        {
            "gene": expr.columns.astype(str),
            "mean_expression": mean_expr.values,
            "variance": var_expr.values,
            "mean_percentile": mean_pct.values,
            "variance_percentile": var_pct.values,
        }
    ).set_index("gene")
    stats["top5_mean"] = stats["mean_percentile"] >= cutoff
    stats["top5_variance"] = stats["variance_percentile"] >= cutoff
    stats["top5_either"] = stats["top5_mean"] | stats["top5_variance"]
    stats["top5_both"] = stats["top5_mean"] & stats["top5_variance"]
    stats["panel_score"] = (
        0.55 * stats["variance_percentile"]
        + 0.35 * stats["mean_percentile"]
        + 0.10 * stats["top5_both"].astype(float)
    )

    candidates = stats[stats["top5_either"]].copy()
    if candidates.shape[0] < n_genes:
        candidates = stats.copy()
    candidates = candidates.sort_values(
        ["top5_both", "panel_score", "variance_percentile", "mean_percentile"],
        ascending=[False, False, False, False],
    )
    genes = list(candidates.head(n_genes).index)
    return genes, stats.loc[genes].reset_index()


def fit_gene_panel_sequences(
    train: ExpressionDataset,
    test: ExpressionDataset,
    host_genes: list[str],
    parasite_genes: list[str],
    time_grid: np.ndarray,
) -> tuple[ModuleSequences, ModuleSequences]:
    host_genes = [g for g in host_genes if g in train.host.columns and g in test.host.columns]
    parasite_genes = [g for g in parasite_genes if g in train.parasite.columns and g in test.parasite.columns]
    if not host_genes or not parasite_genes:
        raise ValueError("Selected gene panel is empty after train/test alignment.")

    host_scaler = StandardScaler()
    parasite_scaler = StandardScaler()
    host_train_scaled = host_scaler.fit_transform(train.host.loc[:, host_genes])
    host_test_scaled = host_scaler.transform(test.host.loc[:, host_genes])
    parasite_train_scaled = parasite_scaler.fit_transform(train.parasite.loc[:, parasite_genes])
    parasite_test_scaled = parasite_scaler.transform(test.parasite.loc[:, parasite_genes])

    train_animals, host_train_seq = interpolate_by_animal(train.meta, host_train_scaled, time_grid)
    _, parasite_train_seq = interpolate_by_animal(train.meta, parasite_train_scaled, time_grid)
    test_animals, host_test_seq = interpolate_by_animal(test.meta, host_test_scaled, time_grid)
    _, parasite_test_seq = interpolate_by_animal(test.meta, parasite_test_scaled, time_grid)

    host_seq_scaler = Standardizer.fit(host_train_seq)
    parasite_seq_scaler = Standardizer.fit(parasite_train_seq)
    host_train_seq = host_seq_scaler.transform(host_train_seq)
    host_test_seq = host_seq_scaler.transform(host_test_seq)
    parasite_train_seq = parasite_seq_scaler.transform(parasite_train_seq)
    parasite_test_seq = parasite_seq_scaler.transform(parasite_test_seq)

    return (
        ModuleSequences(train.experiment, train_animals, time_grid, host_train_seq, parasite_train_seq),
        ModuleSequences(test.experiment, test_animals, time_grid, host_test_seq, parasite_test_seq),
    )


def apidopnetwork_design(host: np.ndarray, time: np.ndarray) -> np.ndarray:
    """First-harmonic first-order APidopnetwork design.

    The fitted observation map has the form

        P(t) = B_s [H(t) sin(omega t)] + B_c [H(t) cos(omega t)]
               + a sin(omega t) + b cos(omega t).

    This is the data-augmentation observation model described in the
    manuscript: host gene coordinates modulate the parasite rhythmic
    amplitudes and phases.
    """
    n_seq, n_time, _ = host.shape
    sin_t = np.sin(OMEGA * time)[None, :, None]
    cos_t = np.cos(OMEGA * time)[None, :, None]
    sin_all = np.tile(sin_t, (n_seq, 1, 1))
    cos_all = np.tile(cos_t, (n_seq, 1, 1))
    return np.concatenate([host * sin_all, host * cos_all, sin_all, cos_all], axis=2)


def fit_apidopnetwork(
    host: np.ndarray,
    parasite: np.ndarray,
    time: np.ndarray,
    regressor: str = "ridge",
    alpha: float | None = None,
    l1_ratio: float = 0.5,
) -> tuple[object, np.ndarray]:
    design = apidopnetwork_design(host, time).reshape(-1, 2 * host.shape[-1] + 2)
    target = parasite.reshape(-1, parasite.shape[-1])
    if regressor == "elasticnet":
        model = MultiTaskElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            alphas=ELASTIC_ALPHAS,
            cv=min(5, design.shape[0]),
            fit_intercept=False,
            max_iter=50000,
            random_state=0,
        )
    elif regressor == "lasso":
        model = MultiTaskLassoCV(
            alphas=LASSO_ALPHAS,
            cv=min(5, design.shape[0]),
            fit_intercept=False,
            max_iter=100000,
            random_state=0,
        )
    elif regressor == "fixed_lasso":
        if alpha is None:
            raise ValueError("alpha must be provided for fixed_lasso")
        model = MultiTaskLasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=100000,
            random_state=0,
        )
    elif regressor == "fixed_elasticnet":
        if alpha is None:
            raise ValueError("alpha must be provided for fixed_elasticnet")
        model = MultiTaskElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=False,
            max_iter=100000,
            random_state=0,
        )
    elif regressor == "ridge":
        model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False)
    else:
        raise ValueError(regressor)
    model.fit(design, target)
    residual = target - model.predict(design)
    cov = np.cov(residual, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-5 * np.eye(parasite.shape[-1])
    return model, cov


def predict_apidopnetwork(model: object, host: np.ndarray, time: np.ndarray) -> np.ndarray:
    design = apidopnetwork_design(host, time).reshape(-1, 2 * host.shape[-1] + 2)
    pred = model.predict(design)
    return pred.reshape(host.shape[0], host.shape[1], -1)


def regenerate_host_gaussian(
    rng: np.random.Generator,
    host_train: np.ndarray,
    n_generated: int,
    shrinkage: float,
) -> np.ndarray:
    """Regenerate complete host trajectories from a shrinkage Gaussian law."""
    n_seq, n_time, d_host = host_train.shape
    flat = host_train.reshape(n_seq, n_time * d_host)
    mean = flat.mean(axis=0)
    cov = np.cov(flat, rowvar=False)
    cov = np.atleast_2d(cov)
    diag = np.diag(np.diag(cov))
    cov = (1.0 - shrinkage) * cov + shrinkage * diag
    cov += 1e-5 * np.eye(cov.shape[0])
    generated = rng.multivariate_normal(mean, cov, size=n_generated)
    return generated.reshape(n_generated, n_time, d_host)


def make_training_pair_panel(
    train_mod: ModuleSequences,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if args.training_set == "original":
        return split_target_condition(train_mod, args.target)

    rng = np.random.default_rng(args.seed + 1201)
    ap_model, residual_cov = fit_apidopnetwork(train_mod.host, train_mod.parasite, train_mod.time_grid)
    host_gen = regenerate_host_gaussian(rng, train_mod.host, args.n_generated, args.host_gaussian_shrinkage)
    parasite_gen = predict_apidopnetwork(ap_model, host_gen, train_mod.time_grid)
    residual = rng.multivariate_normal(
        np.zeros(train_mod.parasite.shape[-1]),
        residual_cov,
        size=args.n_generated * len(train_mod.time_grid),
    ).reshape(args.n_generated, len(train_mod.time_grid), train_mod.parasite.shape[-1])
    parasite_gen = parasite_gen + residual

    host_aug = np.concatenate([train_mod.host, host_gen], axis=0)
    parasite_aug = np.concatenate([train_mod.parasite, parasite_gen], axis=0)
    aug_mod = ModuleSequences(train_mod.experiment, train_mod.animals, train_mod.time_grid, host_aug, parasite_aug)
    return split_target_condition(aug_mod, args.target)


def evaluate_fold_panel(
    train_mod: ModuleSequences,
    test_mod: ModuleSequences,
    fold: str,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    target_train, condition_train = make_training_pair_panel(train_mod, args)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mahpic_geo"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mahpic_gene_panel_input_ablation_e23"))
    parser.add_argument("--target", choices=["parasite", "host"], default="host")
    parser.add_argument("--training-set", choices=["apidopnetwork", "original"], default="apidopnetwork")
    parser.add_argument("--n-panel-genes", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--n-generated", type=int, default=160)
    parser.add_argument("--host-gaussian-shrinkage", type=float, default=0.35)
    parser.add_argument("--train-windows", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--cells", nargs="+", default=["RNN", "LSTM", "GRU"], choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--hidden-dim", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-grid", type=float, nargs="+", default=list(DEFAULT_TIME_GRID))
    args = parser.parse_args()

    table_dir, fig_dir = ensure_dirs(args.out_dir)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    e23 = load_dataset("E23", args.data_dir, cache_dir)
    time_grid = np.asarray(args.time_grid, dtype=float)

    host_genes, host_stats = select_gene_panel(e23.host, args.n_panel_genes, args.top_fraction)
    parasite_genes, parasite_stats = select_gene_panel(e23.parasite, args.n_panel_genes, args.top_fraction)
    host_stats.to_csv(table_dir / "selected_host_gene_panel.csv", index=False)
    parasite_stats.to_csv(table_dir / "selected_parasite_gene_panel.csv", index=False)

    rows: list[dict[str, float | int | str]] = []
    for animal in sorted(e23.meta["animal"].unique()):
        train = subset_dataset(e23, e23.meta["animal"] != animal)
        test = subset_dataset(e23, e23.meta["animal"] == animal)
        train_mod, test_mod = fit_gene_panel_sequences(train, test, host_genes, parasite_genes, time_grid)
        rows.extend(evaluate_fold_panel(train_mod, test_mod, str(animal), args))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(table_dir / "mahpic_gene_panel_input_ablation_raw.csv", index=False)
    summary.to_csv(table_dir / "mahpic_gene_panel_input_ablation_summary.csv", index=False)
    write_tables(summary, table_dir)
    plot_summary(summary, fig_dir)

    metadata = {
        "target": args.target,
        "training_set": args.training_set,
        "feature_mode": "raw_gene_panel",
        "selection_scope": "all E23 animals, unsupervised expression/variance filter",
        "apidopnetwork_model": "P(t)=B_s[H(t)sin(omega t)]+B_c[H(t)cos(omega t)]+a sin(omega t)+b cos(omega t)",
        "host_regeneration": "complete host gene trajectories sampled from a shrinkage multivariate Gaussian distribution",
        "host_gaussian_shrinkage": args.host_gaussian_shrinkage,
        "n_panel_genes_per_species": args.n_panel_genes,
        "top_fraction": args.top_fraction,
        "host_genes": host_genes,
        "parasite_genes": parasite_genes,
        "time_grid": [float(x) for x in time_grid],
        "n_generated": args.n_generated,
        "note": (
            "Selected genes are directly used as coordinates; no PCA module "
            "compression is applied in this experiment."
        ),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
