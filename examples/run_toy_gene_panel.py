#!/usr/bin/env python3
"""Small toy example for the FiNN/APidopnetwork validation workflow.

This script is intentionally lightweight. It generates synthetic paired
host--parasite trajectories and runs a short regression-style smoke test. It is
not intended to reproduce the manuscript numbers; the full reproduction command
is provided in ``scripts/reproduce_mahpic_e23_gene_panel.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def make_toy_data(seed: int, n_animals: int, n_time: int, d_host: int, d_parasite: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 1.0, n_time)
    phase = 2.0 * np.pi * time
    host = np.zeros((n_animals, n_time, d_host))
    parasite = np.zeros((n_animals, n_time, d_parasite))
    host_to_parasite = rng.normal(0.0, 0.4, size=(d_host, d_parasite))
    for i in range(n_animals):
        animal_shift = rng.normal(0.0, 0.2, size=(1, d_host))
        host[i] = (
            animal_shift
            + rng.normal(0.0, 0.35, size=(1, d_host)) * np.sin(phase[:, None])
            + rng.normal(0.0, 0.25, size=(1, d_host)) * np.cos(phase[:, None])
            + rng.normal(0.0, 0.08, size=(n_time, d_host))
        )
        parasite[i] = (
            0.6 * np.sin(phase[:, None])
            + 0.3 * np.cos(phase[:, None])
            + (host[i] * np.sin(phase[:, None])) @ host_to_parasite
            + rng.normal(0.0, 0.10, size=(n_time, d_parasite))
        )
    return host, parasite


def fit_ap_generator(host: np.ndarray, parasite: np.ndarray, alpha: float = 1.0) -> Ridge:
    n, t, _ = host.shape
    phase = np.linspace(0.0, 2.0 * np.pi, t)
    sin_t = np.sin(phase)[None, :, None]
    cos_t = np.cos(phase)[None, :, None]
    x = np.concatenate([sin_t.repeat(n, 0), cos_t.repeat(n, 0), host * sin_t, host * cos_t], axis=2)
    model = Ridge(alpha=alpha)
    model.fit(x.reshape(n * t, -1), parasite.reshape(n * t, parasite.shape[-1]))
    return model


def generate_pairs(host: np.ndarray, parasite: np.ndarray, n_generated: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, t, d_host = host.shape
    mean_path = host.mean(axis=0)
    flat = (host - mean_path).reshape(n, t * d_host)
    cov = np.cov(flat, rowvar=False) + 0.02 * np.eye(t * d_host)
    synthetic_host = rng.multivariate_normal(mean_path.reshape(-1), cov, size=n_generated).reshape(n_generated, t, d_host)
    model = fit_ap_generator(host, parasite)
    phase = np.linspace(0.0, 2.0 * np.pi, t)
    sin_t = np.sin(phase)[None, :, None]
    cos_t = np.cos(phase)[None, :, None]
    x = np.concatenate(
        [sin_t.repeat(n_generated, 0), cos_t.repeat(n_generated, 0), synthetic_host * sin_t, synthetic_host * cos_t],
        axis=2,
    )
    synthetic_parasite = model.predict(x.reshape(n_generated * t, -1)).reshape(n_generated, t, parasite.shape[-1])
    synthetic_parasite += rng.normal(0.0, 0.08, size=synthetic_parasite.shape)
    return synthetic_host, synthetic_parasite


def make_supervised(host: np.ndarray, parasite: np.ndarray, observed: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for i in range(host.shape[0]):
        for k in range(1, observed):
            x_rows.append(np.concatenate([host[i, k - 1], parasite[i, k]]))
            y_rows.append(host[i, k])
    return np.asarray(x_rows), np.asarray(y_rows)


def evaluate(host: np.ndarray, parasite: np.ndarray, method: str, seed: int) -> float:
    observed = host.shape[1] - 1
    errors = []
    for held_out in range(host.shape[0]):
        train_mask = np.arange(host.shape[0]) != held_out
        h_train, p_train = host[train_mask], parasite[train_mask]
        if method == "ap_regeneration":
            h_gen, p_gen = generate_pairs(h_train, p_train, n_generated=24, seed=seed + held_out)
            h_train = np.concatenate([h_train, h_gen], axis=0)
            p_train = np.concatenate([p_train, p_gen], axis=0)
        x_train, y_train = make_supervised(h_train, p_train, observed)
        x_test, y_test = make_supervised(host[held_out : held_out + 1], parasite[held_out : held_out + 1], observed)
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
        y_train = y_scaler.fit_transform(y_train)
        pred = y_scaler.inverse_transform(Ridge(alpha=1.0).fit(x_train, y_train).predict(x_scaler.transform(x_test)))
        errors.append(float(np.mean((pred - y_test) ** 2)))
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/toy_example"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    host, parasite = make_toy_data(args.seed, n_animals=6, n_time=6, d_host=6, d_parasite=6)
    rows = [
        {"method": "No augmentation", "toy_mse": evaluate(host, parasite, "none", args.seed)},
        {"method": "APidopnetwork-style regeneration", "toy_mse": evaluate(host, parasite, "ap_regeneration", args.seed)},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "toy_summary.csv", index=False)
    print(result.to_string(index=False))
    print(f"\nToy output written to {args.out_dir}")


if __name__ == "__main__":
    main()
