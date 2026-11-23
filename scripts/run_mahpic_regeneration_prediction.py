#!/usr/bin/env python3
"""MaHPIC APidopnetwork-regeneration + FiNN-style prediction experiment.

This script follows the manuscript's empirical logic:

    paired host/parasite RNA trajectories
    -> construct low-dimensional host and parasite module coordinates
    -> fit a harmonic or APidopnetwork observation model on the training animals
    -> regenerate biologically constrained paired training trajectories
    -> train the same recurrent predictor on each augmented training set
    -> forecast held-out future module trajectories from shortened prefixes.

APidopnetwork is used only as a training-data regeneration mechanism here.  It
is not concatenated to the recurrent model as an extra network feature.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
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
from sklearn.decomposition import PCA
from sklearn.linear_model import MultiTaskElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - used only when the active env lacks torch.
    torch = None
    nn = None


OMEGA = 2.0 * np.pi / 48.0
RIDGE_ALPHAS = np.logspace(-4, 3, 30)
ELASTIC_ALPHAS = np.logspace(-3, 1, 24)
DEFAULT_TIME_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])


DATASETS = {
    "E23": {
        "host": "GSE104223_E23RM99FGMmCyDaWB_Mulatta-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
        "parasite": "GSE104223_E23RM99FGMmCyDaWB_Cynomolgi-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
    },
    "E25": {
        "host": "GSE104330_E25bM99FGMmCyDaWB_Mulatta-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
        "parasite": "GSE104330_E25bM99FGMmCyDaWB_Cynomolgi-Genes-RawCounts-Results_MULTIPL_GEO.xlsx",
    },
}


@dataclass
class ExpressionDataset:
    experiment: str
    host: pd.DataFrame
    parasite: pd.DataFrame
    meta: pd.DataFrame


@dataclass
class ModuleSequences:
    experiment: str
    animals: list[str]
    time_grid: np.ndarray
    host: np.ndarray
    parasite: np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        flat = values.reshape(-1, values.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        scale = flat.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


@dataclass
class ObservationModel:
    kind: str
    coef: np.ndarray
    residual_cov: np.ndarray
    d_host: int
    d_parasite: int


class SimpleRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int, lr: float) -> None:
        rng = np.random.default_rng(seed)
        self.wxh = rng.normal(0.0, 0.12, size=(input_dim, hidden_dim))
        self.whh = rng.normal(0.0, 0.08, size=(hidden_dim, hidden_dim))
        self.bh = np.zeros((1, hidden_dim))
        self.why = rng.normal(0.0, 0.12, size=(hidden_dim, output_dim))
        self.by = np.zeros((1, output_dim))
        self.lr = lr
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in self.params().items()}
        self.v = {name: np.zeros_like(value) for name, value in self.params().items()}

    def params(self) -> dict[str, np.ndarray]:
        return {"wxh": self.wxh, "whh": self.whh, "bh": self.bh, "why": self.why, "by": self.by}

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch, n_time, _ = inputs.shape
        hidden_dim = self.whh.shape[0]
        h = np.zeros((batch, n_time + 1, hidden_dim))
        yhat = np.zeros((batch, n_time, self.why.shape[1]))
        for k in range(n_time):
            h[:, k + 1, :] = np.tanh(inputs[:, k, :] @ self.wxh + h[:, k, :] @ self.whh + self.bh)
            yhat[:, k, :] = h[:, k + 1, :] @ self.why + self.by
        return yhat, h

    def backward(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        pred: np.ndarray,
        hidden: np.ndarray,
    ) -> dict[str, np.ndarray]:
        batch, n_time, _ = inputs.shape
        norm = batch * n_time * targets.shape[-1]
        dy = 2.0 * (pred - targets) / norm
        grads = {name: np.zeros_like(value) for name, value in self.params().items()}
        dh_next = np.zeros((batch, self.whh.shape[0]))
        for k in range(n_time - 1, -1, -1):
            grads["why"] += hidden[:, k + 1, :].T @ dy[:, k, :]
            grads["by"] += dy[:, k, :].sum(axis=0, keepdims=True)
            dh = dy[:, k, :] @ self.why.T + dh_next
            da = dh * (1.0 - hidden[:, k + 1, :] ** 2)
            grads["wxh"] += inputs[:, k, :].T @ da
            grads["whh"] += hidden[:, k, :].T @ da
            grads["bh"] += da.sum(axis=0, keepdims=True)
            dh_next = da @ self.whh.T
        for key in grads:
            np.clip(grads[key], -1.0, 1.0, out=grads[key])
        return grads

    def adam_step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in self.params().items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grads[name]
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grads[name] ** 2)
            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int, batch_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n_seq = inputs.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n_seq)
            for start in range(0, n_seq, batch_size):
                idx = order[start : start + batch_size]
                pred, hidden = self.forward(inputs[idx])
                self.adam_step(self.backward(inputs[idx], targets[idx], pred, hidden))

    def step(self, input_t: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_next = np.tanh(input_t @ self.wxh + hidden @ self.whh + self.bh)
        output = hidden_next @ self.why + self.by
        return output, hidden_next


class TorchSequenceModel(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, cell_type: str) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for TorchSequenceModel.")
        super().__init__()
        cell_type = cell_type.upper()
        if cell_type == "RNN":
            self.cell = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif cell_type == "LSTM":
            self.cell = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif cell_type == "GRU":
            self.cell = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown recurrent cell type: {cell_type}")
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: "torch.Tensor", hidden: object | None = None) -> tuple["torch.Tensor", object]:
        out, hidden_next = self.cell(inputs, hidden)  # type: ignore[arg-type]
        return self.readout(out), hidden_next

    def step(self, input_t: "torch.Tensor", hidden: object | None = None) -> tuple["torch.Tensor", object]:
        out, hidden_next = self.cell(input_t[:, None, :], hidden)  # type: ignore[arg-type]
        return self.readout(out[:, 0, :]), hidden_next


def ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    cache_dir = out_dir / "cache"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, fig_dir, cache_dir


def parse_sample_id(value: str) -> tuple[str, int] | None:
    match = re.match(r"WB_([^_]+)_(\d+)[A-Za-z]*$", str(value))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def find_sample_header_row(raw: pd.DataFrame) -> int:
    best_row, best_count = -1, 0
    for i in range(min(len(raw), 80)):
        count = sum(parse_sample_id(v) is not None for v in raw.iloc[i, 2:].tolist())
        if count > best_count:
            best_row, best_count = i, count
    if best_count < 3:
        raise ValueError("Could not locate sample header row.")
    return best_row


def read_expression_matrix(path: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = cache_dir / f"{path.stem}.pkl"
    if cache_path.exists() and cache_path.stat().st_mtime > path.stat().st_mtime:
        cached = pd.read_pickle(cache_path)
        return cached["expr"], cached["meta"]

    raw = pd.read_excel(path, sheet_name="RawReadCounts", header=None)
    header_row = find_sample_header_row(raw)
    sample_ids = [str(v) for v in raw.iloc[header_row, 2:].tolist()]
    parsed = [parse_sample_id(v) for v in sample_ids]
    keep = [i for i, item in enumerate(parsed) if item is not None]
    data = raw.iloc[header_row + 1 :, :]
    gene_ids = data.iloc[:, 0].astype(str)
    valid = gene_ids.notna() & (gene_ids != "nan")
    data = data.loc[valid]
    gene_ids = gene_ids.loc[valid]
    expr = data.iloc[:, [2 + i for i in keep]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expr.index = gene_ids.values
    expr.columns = [sample_ids[i] for i in keep]
    expr = expr.groupby(expr.index).sum().T
    expr = np.log1p(expr)

    meta_rows = []
    for sid in expr.index:
        animal, tp = parse_sample_id(sid)  # type: ignore[misc]
        meta_rows.append({"sample_id": sid, "animal": animal, "timepoint": tp})
    meta = pd.DataFrame(meta_rows).set_index("sample_id")

    key = meta["animal"].astype(str) + "__T" + meta["timepoint"].astype(str)
    expr = expr.groupby(key.values).mean()
    meta = meta.groupby(key.values).first()
    expr.index = meta.index
    pd.to_pickle({"expr": expr, "meta": meta}, cache_path)
    return expr, meta


def load_dataset(name: str, data_dir: Path, cache_dir: Path) -> ExpressionDataset:
    spec = DATASETS[name]
    host_path = data_dir / spec["host"]
    parasite_path = data_dir / spec["parasite"]
    missing = [p for p in [host_path, parasite_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(p) for p in missing))
    host, host_meta = read_expression_matrix(host_path, cache_dir)
    parasite, _ = read_expression_matrix(parasite_path, cache_dir)
    common = host.index.intersection(parasite.index)
    meta = host_meta.loc[common].copy()
    meta["experiment"] = name
    order = meta.sort_values(["animal", "timepoint"]).index
    return ExpressionDataset(
        experiment=name,
        host=host.loc[order],
        parasite=parasite.loc[order],
        meta=meta.loc[order].reset_index(drop=True),
    )


def subset_dataset(ds: ExpressionDataset, mask: np.ndarray | pd.Series) -> ExpressionDataset:
    values = np.asarray(mask)
    return ExpressionDataset(
        experiment=ds.experiment,
        host=ds.host.loc[values],
        parasite=ds.parasite.loc[values],
        meta=ds.meta.loc[values].reset_index(drop=True),
    )


def common_gene_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = train.columns.intersection(test.columns)
    return train.loc[:, cols], test.loc[:, cols]


def top_variable_columns(frame: pd.DataFrame, n: int) -> list[str]:
    return list(frame.var(axis=0).sort_values(ascending=False).head(min(n, frame.shape[1])).index)


def interpolate_by_animal(meta: pd.DataFrame, values: np.ndarray, time_grid: np.ndarray) -> tuple[list[str], np.ndarray]:
    animals = sorted(meta["animal"].unique())
    seq = np.zeros((len(animals), len(time_grid), values.shape[1]))
    for i, animal in enumerate(animals):
        idx = np.where(meta["animal"].values == animal)[0]
        times = meta.iloc[idx]["timepoint"].astype(float).values
        order = np.argsort(times)
        times = times[order]
        vals = values[idx][order]
        for j in range(values.shape[1]):
            seq[i, :, j] = np.interp(time_grid, times, vals[:, j])
    return animals, seq


def fit_module_sequences(
    train: ExpressionDataset,
    test: ExpressionDataset,
    n_host_components: int,
    n_parasite_components: int,
    n_top_genes: int,
    time_grid: np.ndarray,
) -> tuple[ModuleSequences, ModuleSequences]:
    host_train, host_test = common_gene_frames(train.host, test.host)
    parasite_train, parasite_test = common_gene_frames(train.parasite, test.parasite)
    host_genes = top_variable_columns(host_train, n_top_genes)
    parasite_genes = top_variable_columns(parasite_train, n_top_genes)

    host_scaler = StandardScaler()
    parasite_scaler = StandardScaler()
    host_train_scaled = host_scaler.fit_transform(host_train.loc[:, host_genes])
    host_test_scaled = host_scaler.transform(host_test.loc[:, host_genes])
    parasite_train_scaled = parasite_scaler.fit_transform(parasite_train.loc[:, parasite_genes])
    parasite_test_scaled = parasite_scaler.transform(parasite_test.loc[:, parasite_genes])

    h_dim = min(n_host_components, host_train_scaled.shape[0], host_train_scaled.shape[1])
    p_dim = min(n_parasite_components, parasite_train_scaled.shape[0], parasite_train_scaled.shape[1])
    host_pca = PCA(n_components=h_dim, random_state=0)
    parasite_pca = PCA(n_components=p_dim, random_state=0)
    host_train_mod = host_pca.fit_transform(host_train_scaled)
    host_test_mod = host_pca.transform(host_test_scaled)
    parasite_train_mod = parasite_pca.fit_transform(parasite_train_scaled)
    parasite_test_mod = parasite_pca.transform(parasite_test_scaled)

    train_animals, host_train_seq = interpolate_by_animal(train.meta, host_train_mod, time_grid)
    _, parasite_train_seq = interpolate_by_animal(train.meta, parasite_train_mod, time_grid)
    test_animals, host_test_seq = interpolate_by_animal(test.meta, host_test_mod, time_grid)
    _, parasite_test_seq = interpolate_by_animal(test.meta, parasite_test_mod, time_grid)

    host_std = Standardizer.fit(host_train_seq)
    parasite_std = Standardizer.fit(parasite_train_seq)
    host_train_seq = host_std.transform(host_train_seq)
    host_test_seq = host_std.transform(host_test_seq)
    parasite_train_seq = parasite_std.transform(parasite_train_seq)
    parasite_test_seq = parasite_std.transform(parasite_test_seq)

    return (
        ModuleSequences(train.experiment, train_animals, time_grid, host_train_seq, parasite_train_seq),
        ModuleSequences(test.experiment, test_animals, time_grid, host_test_seq, parasite_test_seq),
    )


def observation_features(host: np.ndarray, time: np.ndarray, kind: str) -> np.ndarray:
    n_seq, n_time, d_host = host.shape
    sin_t = np.sin(OMEGA * time)[None, :, None]
    cos_t = np.cos(OMEGA * time)[None, :, None]
    ones = np.ones((n_seq, n_time, 1))
    sin_all = np.tile(sin_t, (n_seq, 1, 1))
    cos_all = np.tile(cos_t, (n_seq, 1, 1))
    if kind == "harmonic":
        return np.concatenate([ones, sin_all, cos_all], axis=2)
    if kind == "ap":
        return np.concatenate([ones, sin_all, cos_all, host * sin_all, host * cos_all], axis=2)
    raise ValueError(kind)


def fit_observation_model(host: np.ndarray, parasite: np.ndarray, time: np.ndarray, kind: str) -> ObservationModel:
    design = observation_features(host, time, kind).reshape(-1, observation_features(host, time, kind).shape[-1])
    target = parasite.reshape(-1, parasite.shape[-1])
    if kind == "ap" and design.shape[0] >= 10:
        try:
            model = MultiTaskElasticNetCV(
                l1_ratio=[0.5],
                alphas=ELASTIC_ALPHAS,
                cv=min(5, design.shape[0]),
                max_iter=30000,
                random_state=0,
            )
            model.fit(design, target)
            coef = np.asarray(model.coef_, dtype=float).T
            intercept = np.asarray(model.intercept_, dtype=float)[None, :]
            pred = design @ coef + intercept
            coef = np.vstack([intercept, coef])
            design_for_pred = np.concatenate([np.ones((design.shape[0], 1)), design], axis=1)
            residual = target - design_for_pred @ coef
            return ObservationModel(kind=f"{kind}_intercept", coef=coef, residual_cov=residual_cov(residual), d_host=host.shape[-1], d_parasite=parasite.shape[-1])
        except Exception:
            pass
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=False)
    ridge.fit(design, target)
    pred = ridge.predict(design)
    return ObservationModel(kind=kind, coef=np.asarray(ridge.coef_, dtype=float).T, residual_cov=residual_cov(target - pred), d_host=host.shape[-1], d_parasite=parasite.shape[-1])


def residual_cov(residual: np.ndarray) -> np.ndarray:
    cov = np.cov(residual, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += 1e-5 * np.eye(cov.shape[0])
    return cov


def predict_observation(model: ObservationModel, host: np.ndarray, time: np.ndarray) -> np.ndarray:
    kind = model.kind.replace("_intercept", "")
    features = observation_features(host, time, kind)
    flat = features.reshape(-1, features.shape[-1])
    if model.kind.endswith("_intercept"):
        flat = np.concatenate([np.ones((flat.shape[0], 1)), flat], axis=1)
    pred = flat @ model.coef
    return pred.reshape(host.shape[0], host.shape[1], model.d_parasite)


def generate_host_trajectories(rng: np.random.Generator, host_train: np.ndarray, n_generated: int) -> np.ndarray:
    n_train, n_time, d_host = host_train.shape
    mean = host_train.mean(axis=0)
    residual = host_train - mean[None, :, :]
    flat_resid = residual.reshape(-1, d_host)
    cov = np.cov(flat_resid, rowvar=False) + 1e-5 * np.eye(d_host)
    generated = np.zeros((n_generated, n_time, d_host))
    phase = np.linspace(0.0, 2.0 * np.pi, n_time)
    for i in range(n_generated):
        base = host_train[rng.integers(0, n_train)].copy()
        amp = rng.lognormal(mean=0.0, sigma=0.12, size=(1, d_host))
        offset = rng.normal(0.0, 0.08, size=(1, d_host))
        smooth = np.zeros_like(base)
        for j in range(d_host):
            smooth[:, j] = (
                rng.normal(0.0, 0.035) * np.sin(phase + rng.uniform(-0.5, 0.5))
                + rng.normal(0.0, 0.025) * np.cos(2.0 * phase + rng.uniform(-0.5, 0.5))
            )
        noise = rng.multivariate_normal(np.zeros(d_host), 0.16 * cov, size=n_time)
        generated[i] = mean + amp * (base - mean) + offset + smooth + noise
    return generated


def make_regenerated_sequences(
    rng: np.random.Generator,
    host_train: np.ndarray,
    parasite_train: np.ndarray,
    time: np.ndarray,
    kind: str,
    n_generated: int,
) -> np.ndarray:
    if kind == "bootstrap":
        idx = rng.integers(0, host_train.shape[0], size=n_generated)
        host_boot = host_train[idx] + 0.05 * rng.normal(size=(n_generated, host_train.shape[1], host_train.shape[2]))
        parasite_boot = parasite_train[idx] + 0.05 * rng.normal(
            size=(n_generated, parasite_train.shape[1], parasite_train.shape[2])
        )
        return np.concatenate([host_boot, parasite_boot], axis=2)
    host_gen = generate_host_trajectories(rng, host_train, n_generated)
    obs_kind = "harmonic" if kind == "harmonic" else "ap"
    model = fit_observation_model(host_train, parasite_train, time, obs_kind)
    parasite_gen = predict_observation(model, host_gen, time)
    residual = rng.multivariate_normal(
        np.zeros(model.d_parasite),
        model.residual_cov,
        size=host_gen.shape[0] * host_gen.shape[1],
    ).reshape(host_gen.shape[0], host_gen.shape[1], model.d_parasite)
    return np.concatenate([host_gen, parasite_gen + residual], axis=2)


def add_time_features(seq: np.ndarray, time: np.ndarray) -> np.ndarray:
    n_seq, _, _ = seq.shape
    t_scaled = (time / max(float(time.max()), 1.0))[None, :, None]
    sin_t = np.sin(OMEGA * time)[None, :, None]
    cos_t = np.cos(OMEGA * time)[None, :, None]
    features = np.tile(np.concatenate([t_scaled, sin_t, cos_t], axis=2), (n_seq, 1, 1))
    return np.concatenate([seq, features], axis=2)


def make_transition_data(seq: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return add_time_features(seq[:, :-1, :], time[:-1]), seq[:, 1:, :]


def fit_predictor(
    train_seq: np.ndarray,
    time: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[SimpleRNN, Standardizer, Standardizer]:
    train_inputs, train_targets = make_transition_data(train_seq, time)
    input_scaler = Standardizer.fit(train_inputs)
    target_scaler = Standardizer.fit(train_targets)
    x = input_scaler.transform(train_inputs)
    y = target_scaler.transform(train_targets)
    model = SimpleRNN(x.shape[-1], hidden_dim, y.shape[-1], seed=seed, lr=lr)
    model.train(x, y, epochs=epochs, batch_size=batch_size, seed=seed + 501)
    return model, input_scaler, target_scaler


def fit_predictor_torch(
    train_seq: np.ndarray,
    time: np.ndarray,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    cell_type: str,
) -> tuple[TorchSequenceModel, Standardizer, Standardizer]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available in the active Python environment. "
            "Run with a Python environment in which PyTorch is installed."
        )
    torch.manual_seed(seed)
    train_inputs, train_targets = make_transition_data(train_seq, time)
    input_scaler = Standardizer.fit(train_inputs)
    target_scaler = Standardizer.fit(train_targets)
    x = input_scaler.transform(train_inputs).astype(np.float32)
    y = target_scaler.transform(train_targets).astype(np.float32)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    model = TorchSequenceModel(x.shape[-1], hidden_dim, y.shape[-1], cell_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # type: ignore[union-attr]
    n_seq = x.shape[0]
    rng = np.random.default_rng(seed + 501)
    for _ in range(epochs):
        order = rng.permutation(n_seq)
        for start in range(0, n_seq, batch_size):
            idx = order[start : start + batch_size]
            pred, _ = model(xt[idx])
            loss = loss_fn(pred, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
    return model, input_scaler, target_scaler


def forecast_future(
    model: SimpleRNN,
    input_scaler: Standardizer,
    target_scaler: Standardizer,
    observed_seq: np.ndarray,
    time: np.ndarray,
    train_window: int,
) -> np.ndarray:
    n_seq, n_time, _ = observed_seq.shape
    hidden = np.zeros((n_seq, model.whh.shape[0]))
    current = observed_seq[:, 0, :]
    for k in range(train_window - 1):
        inp = add_time_features(current[:, None, :], time[k : k + 1])[:, 0, :]
        _, hidden = model.step(input_scaler.transform(inp[:, None, :])[:, 0, :], hidden)
        current = observed_seq[:, k + 1, :]

    pred_seq = observed_seq.copy()
    current = observed_seq[:, train_window - 1, :]
    for k in range(train_window - 1, n_time - 1):
        inp = add_time_features(current[:, None, :], time[k : k + 1])[:, 0, :]
        pred_scaled, hidden = model.step(input_scaler.transform(inp[:, None, :])[:, 0, :], hidden)
        pred = target_scaler.inverse(pred_scaled[:, None, :])[:, 0, :]
        pred_seq[:, k + 1, :] = pred
        current = pred
    return pred_seq


def forecast_future_torch(
    model: TorchSequenceModel,
    input_scaler: Standardizer,
    target_scaler: Standardizer,
    observed_seq: np.ndarray,
    time: np.ndarray,
    train_window: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is not available in the active Python environment.")
    model.eval()
    n_seq, n_time, _ = observed_seq.shape
    pred_seq = observed_seq.copy()
    hidden: object | None = None
    current = observed_seq[:, 0, :]
    with torch.no_grad():
        for k in range(train_window - 1):
            inp = add_time_features(current[:, None, :], time[k : k + 1])[:, 0, :]
            inp_scaled = input_scaler.transform(inp[:, None, :])[:, 0, :].astype(np.float32)
            _, hidden = model.step(torch.tensor(inp_scaled), hidden)
            current = observed_seq[:, k + 1, :]

        current = observed_seq[:, train_window - 1, :]
        for k in range(train_window - 1, n_time - 1):
            inp = add_time_features(current[:, None, :], time[k : k + 1])[:, 0, :]
            inp_scaled = input_scaler.transform(inp[:, None, :])[:, 0, :].astype(np.float32)
            pred_scaled, hidden = model.step(torch.tensor(inp_scaled), hidden)
            pred = target_scaler.inverse(pred_scaled.numpy()[:, None, :])[:, 0, :]
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


def evaluate_sequences(
    train_mod: ModuleSequences,
    test_mod: ModuleSequences,
    split: str,
    fold: str,
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    time = train_mod.time_grid
    train_seq = np.concatenate([train_mod.host, train_mod.parasite], axis=2)
    test_seq = np.concatenate([test_mod.host, test_mod.parasite], axis=2)
    methods = {
        "No augmentation": train_seq,
        "Bootstrap augmentation": np.concatenate(
            [
                train_seq,
                make_regenerated_sequences(
                    np.random.default_rng(args.seed + 1001 + len(fold)),
                    train_mod.host,
                    train_mod.parasite,
                    time,
                    "bootstrap",
                    args.n_generated,
                ),
            ],
            axis=0,
        ),
        "Harmonic regeneration": np.concatenate(
            [
                train_seq,
                make_regenerated_sequences(
                    np.random.default_rng(args.seed + 1101 + len(fold)),
                    train_mod.host,
                    train_mod.parasite,
                    time,
                    "harmonic",
                    args.n_generated,
                ),
            ],
            axis=0,
        ),
        "APidopnetwork regeneration": np.concatenate(
            [
                train_seq,
                make_regenerated_sequences(
                    np.random.default_rng(args.seed + 1201 + len(fold)),
                    train_mod.host,
                    train_mod.parasite,
                    time,
                    "ap",
                    args.n_generated,
                ),
            ],
            axis=0,
        ),
    }

    rows: list[dict[str, float | int | str]] = []
    for train_window in args.train_windows:
        if train_window >= len(time):
            continue
        future_slice = slice(train_window, len(time))
        for method, method_train in methods.items():
            for cell_type in args.cells:
                model, input_scaler, target_scaler = fit_predictor_torch(
                    method_train,
                    time,
                    seed=args.seed + 3000 + train_window + len(method) + len(fold) + 37 * len(cell_type),
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    cell_type=cell_type,
                )
                pred = forecast_future_torch(model, input_scaler, target_scaler, test_seq, time, train_window)
                true_future = test_seq[:, future_slice, :]
                pred_future = pred[:, future_slice, :]
                d_host = train_mod.host.shape[-1]
                total_mse, total_r2 = mse_r2(true_future, pred_future)
                host_mse, host_r2 = mse_r2(true_future[:, :, :d_host], pred_future[:, :, :d_host])
                parasite_mse, parasite_r2 = mse_r2(true_future[:, :, d_host:], pred_future[:, :, d_host:])
                rows.append(
                    {
                        "split": split,
                        "fold": fold,
                        "method": method,
                        "cell_type": cell_type.upper(),
                        "observed_time_points": int(train_window),
                        "observed_until": float(time[train_window - 1]),
                        "future_time_points": int(len(time) - train_window),
                        "n_original_train_trajectories": int(train_seq.shape[0]),
                        "n_generated_trajectories": int(0 if method == "No augmentation" else args.n_generated),
                        "n_total_train_trajectories": int(method_train.shape[0]),
                        "total_mse": total_mse,
                        "host_mse": host_mse,
                        "parasite_mse": parasite_mse,
                        "total_r2": total_r2,
                        "host_r2": host_r2,
                        "parasite_r2": parasite_r2,
                    }
                )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["split", "observed_time_points", "observed_until", "method", "cell_type"], sort=False)
        .agg(
            total_mse_mean=("total_mse", "mean"),
            total_mse_sd=("total_mse", "std"),
            host_mse_mean=("host_mse", "mean"),
            host_mse_sd=("host_mse", "std"),
            parasite_mse_mean=("parasite_mse", "mean"),
            parasite_mse_sd=("parasite_mse", "std"),
            total_r2_mean=("total_r2", "mean"),
            total_r2_sd=("total_r2", "std"),
            n_folds=("fold", "count"),
            original_train_trajectories=("n_original_train_trajectories", "median"),
            generated_trajectories=("n_generated_trajectories", "median"),
        )
        .reset_index()
    )


def fmt_cell(mean: float, sd: float) -> str:
    if np.isnan(sd):
        sd = 0.0
    return f"{mean:.4f} ({sd:.4f})"


def write_regeneration_table(summary: pd.DataFrame, table_dir: Path) -> None:
    methods = ["No augmentation", "Bootstrap augmentation", "Harmonic regeneration", "APidopnetwork regeneration"]
    headers = ["Split/window", "No aug.", "Bootstrap", "Harmonic regen.", "APidopnetwork regen."]
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for (split, window, until, cell_type), group in summary.groupby(
        ["split", "observed_time_points", "observed_until", "cell_type"], sort=False
    ):
        cells, values = [], []
        for method in methods:
            row = group[group["method"] == method].iloc[0]
            value = float(row["total_mse_mean"])
            values.append(value)
            cells.append(fmt_cell(value, float(row["total_mse_sd"])))
        best = int(np.argmin(values))
        cells[best] = f"\\textbf{{{cells[best]}}}"
        label = f"{split}, {cell_type}, first {int(window)} TP"
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / "mahpic_regeneration_prediction_total_mse_table.tex").write_text("\n".join(lines) + "\n")


def write_finn_style_table(summary: pd.DataFrame, table_dir: Path) -> None:
    method_to_type = {
        "APidopnetwork regeneration": "FiNN / APidopnetwork regeneration",
        "No augmentation": "No-augmentation recurrent",
        "Harmonic regeneration": "Harmonic-regeneration recurrent",
        "Bootstrap augmentation": "Bootstrap-regeneration recurrent",
    }
    for split, split_frame in summary.groupby("split", sort=False):
        windows = sorted(split_frame["observed_time_points"].unique(), reverse=True)
        lines = [
            "\\begin{tabular}{ll" + "c" * len(windows) + "}",
            "\\toprule",
            "Type & Model & " + " & ".join([f"first {int(w)} TP" for w in windows]) + " \\\\",
            "\\midrule",
        ]
        for method in ["APidopnetwork regeneration", "No augmentation", "Harmonic regeneration", "Bootstrap augmentation"]:
            for cell_type in ["RNN", "LSTM", "GRU"]:
                row_filter = (split_frame["method"] == method) & (split_frame["cell_type"] == cell_type)
                if not row_filter.any():
                    continue
                cells = []
                for window in windows:
                    row = split_frame[row_filter & (split_frame["observed_time_points"] == window)].iloc[0]
                    cells.append(f"{float(row['total_mse_mean']):.4f}")
                lines.append(f"{method_to_type[method]} & {cell_type} & " + " & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        safe = re.sub(r"[^A-Za-z0-9]+", "_", split).strip("_")
        (table_dir / f"mahpic_finn_style_mse_table_{safe}.tex").write_text("\n".join(lines) + "\n")


def write_sample_size_table(summary: pd.DataFrame, table_dir: Path) -> None:
    rows = (
        summary.groupby(["split", "observed_time_points"], sort=False)
        .agg(
            original=("original_train_trajectories", "median"),
            generated=("generated_trajectories", "max"),
        )
        .reset_index()
    )
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Split/window & Original trajectories & Generated trajectories & Total augmented trajectories \\\\",
        "\\midrule",
    ]
    for _, row in rows.iterrows():
        total = int(row["original"] + row["generated"])
        lines.append(
            f"{row['split']}, first {int(row['observed_time_points'])} TP & "
            f"{int(row['original'])} & {int(row['generated'])} & {total} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (table_dir / "mahpic_regeneration_sample_size_table.tex").write_text("\n".join(lines) + "\n")


def plot_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    colors = {
        "No augmentation": "#4C78A8",
        "Bootstrap augmentation": "#7A869A",
        "Harmonic regeneration": "#F2A541",
        "APidopnetwork regeneration": "#E45756",
    }
    for (split, cell_type), frame in summary.groupby(["split", "cell_type"], sort=False):
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        windows = sorted(frame["observed_time_points"].unique(), reverse=True)
        for method, method_frame in frame.groupby("method", sort=False):
            method_frame = method_frame.set_index("observed_time_points").loc[windows]
            ax.errorbar(
                windows,
                method_frame["total_mse_mean"],
                yerr=method_frame["total_mse_sd"].fillna(0.0),
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=method,
                color=colors.get(method, None),
            )
        ax.set_xlabel("Observed prefix length (time points)")
        ax.set_ylabel("Future trajectory MSE")
        ax.invert_xaxis()
        ax.grid(True, color="#E6E6E6")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"{split} ({cell_type})")
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{split}_{cell_type}").strip("_")
        fig.savefig(fig_dir / f"mahpic_regeneration_prediction_{safe}.png", dpi=240)
        plt.close(fig)


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_dir, fig_dir, cache_dir = ensure_dirs(args.out_dir)
    e23 = load_dataset("E23", args.data_dir, cache_dir)
    time_grid = np.asarray(args.time_grid, dtype=float)

    rows: list[dict[str, float | int | str]] = []
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
        rows.extend(evaluate_sequences(train_mod, test_mod, "E23 leave-one-animal-out", str(animal), args))

    if args.include_external:
        e25 = load_dataset("E25", args.data_dir, cache_dir)
        train_mod, e25_mod = fit_module_sequences(
            e23,
            e25,
            args.n_host_components,
            args.n_parasite_components,
            args.n_top_genes,
            time_grid,
        )
        for i, animal in enumerate(e25_mod.animals):
            test_mod = ModuleSequences(
                experiment=e25_mod.experiment,
                animals=[animal],
                time_grid=time_grid,
                host=e25_mod.host[i : i + 1],
                parasite=e25_mod.parasite[i : i + 1],
            )
            rows.extend(evaluate_sequences(train_mod, test_mod, "External E23 -> E25", animal, args))

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(table_dir / "mahpic_regeneration_prediction_raw.csv", index=False)
    summary.to_csv(table_dir / "mahpic_regeneration_prediction_summary.csv", index=False)
    write_regeneration_table(summary, table_dir)
    write_finn_style_table(summary, table_dir)
    write_sample_size_table(summary, table_dir)
    plot_summary(summary, fig_dir)
    metadata = {
        "data_dir": str(args.data_dir),
        "time_grid": [float(x) for x in time_grid],
        "n_host_components": args.n_host_components,
        "n_parasite_components": args.n_parasite_components,
        "n_top_genes": args.n_top_genes,
        "n_generated": args.n_generated,
        "train_windows": args.train_windows,
        "cells": args.cells,
        "epochs": args.epochs,
        "include_external": bool(args.include_external),
        "note": "APidopnetwork is used for training-data regeneration, not as an extra RNN input feature.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mahpic_geo"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mahpic_regeneration_prediction"))
    parser.add_argument("--n-host-components", type=int, default=5)
    parser.add_argument("--n-parasite-components", type=int, default=5)
    parser.add_argument("--n-top-genes", type=int, default=500)
    parser.add_argument("--n-generated", type=int, default=160)
    parser.add_argument("--train-windows", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--cells", nargs="+", default=["RNN", "LSTM", "GRU"], choices=["RNN", "LSTM", "GRU"])
    parser.add_argument("--hidden-dim", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=650)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--time-grid", type=float, nargs="+", default=list(DEFAULT_TIME_GRID))
    parser.add_argument("--include-external", action="store_true", help="Also run the external E23-to-E25 transfer test.")
    args = parser.parse_args()

    _, summary = run_experiment(args)
    print(summary.to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
