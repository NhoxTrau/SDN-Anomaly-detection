from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .common import (
    DEFAULT_AE_PREPARED_DIR,
    DEFAULT_FEATURE_SCHEME,
    DEFAULT_INSDN_DIR,
    RANDOM_SEED,
    feature_names_for_scheme,
    resolve_project_path,
)
from .insdn_loader import load_insdn_dataframe
from .prepare_data import split_grouped_sequences_pre_window

DEFAULT_SEQ_LEN = 4
DEFAULT_STRIDE = 1
DEFAULT_MAX_TRAIN_NORMAL_ROWS = 300_000
DEFAULT_SPLIT_GAP = 1
N_QUANTILES = 1000


def _filter_train_normals(split: dict[str, object]) -> dict[str, object]:
    train_mask = np.asarray(split["y_binary_train"], dtype=np.int64) == 0
    if not np.any(train_mask):
        raise ValueError("AE train split is empty after keeping only normal samples.")
    for key in ("X_train", "y_binary_train", "y_multi_train"):
        split[key] = np.asarray(split[key])[train_mask]
    meta_train = split.get("meta_train")
    if isinstance(meta_train, pd.DataFrame):
        split["meta_train"] = meta_train.iloc[np.where(train_mask)[0]].reset_index(drop=True)
    return split


def _cap_train_normals(split: dict[str, object], max_rows: int | None) -> dict[str, object]:
    if max_rows is None or max_rows <= 0:
        return split
    n_rows = len(split["X_train"])
    if n_rows <= max_rows:
        return split
    rng = np.random.default_rng(RANDOM_SEED)
    keep = np.sort(rng.choice(n_rows, size=max_rows, replace=False))
    for key in ("X_train", "y_binary_train", "y_multi_train"):
        split[key] = np.asarray(split[key])[keep]
    meta_train = split.get("meta_train")
    if isinstance(meta_train, pd.DataFrame):
        split["meta_train"] = meta_train.iloc[keep].reset_index(drop=True)
    return split


def load_prepared_v2(prepared_dir: str | Path) -> dict[str, object]:
    prepared_path = resolve_project_path(prepared_dir)
    dataset: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        payload = np.load(prepared_path / f"{split_name}.npz")
        dataset[f"X_{split_name}"] = payload["X"].astype(np.float32)
        dataset[f"y_binary_{split_name}"] = payload["y_binary"].astype(np.int64)
        dataset[f"y_multi_{split_name}"] = payload["y_multi"].astype(np.int64)
        meta_path = prepared_path / f"{split_name}_metadata.csv"
        dataset[f"meta_{split_name}"] = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    with (prepared_path / "scaler_v2.pkl").open("rb") as handle:
        dataset["scaler"] = pickle.load(handle)
    with (prepared_path / "config.json").open("r", encoding="utf-8") as handle:
        dataset["config"] = json.load(handle)
    return dataset


def fit_and_transform_v2(split: dict[str, object], feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> dict[str, object]:
    feature_cols = feature_names_for_scheme(feature_scheme)
    n_features = len(feature_cols)
    X_train = np.asarray(split["X_train"], dtype=np.float32)
    train_flat = X_train.reshape(-1, n_features) if len(X_train) else np.zeros((1, n_features), dtype=np.float32)
    train_flat = np.nan_to_num(train_flat, nan=0.0, posinf=1e6, neginf=-1e6)

    pipeline = Pipeline([
        (
            "quantile",
            QuantileTransformer(
                n_quantiles=min(N_QUANTILES, len(train_flat)),
                output_distribution="uniform",
                random_state=RANDOM_SEED,
                copy=True,
            ),
        ),
        ("standard", StandardScaler()),
    ])
    pipeline.fit(train_flat)

    for key in ("X_train", "X_val", "X_test"):
        X = np.asarray(split[key], dtype=np.float32)
        if len(X) == 0:
            continue
        orig_shape = X.shape
        X_flat = np.nan_to_num(X.reshape(-1, n_features), nan=0.0, posinf=1e6, neginf=-1e6)
        split[key] = pipeline.transform(X_flat).reshape(orig_shape).astype(np.float32)

    split["scaler"] = pipeline
    split["scaler_type"] = "quantile_uniform_then_standard"
    return split


def save_prepared_v2(dataset: dict[str, object], save_dir: Path, config: dict[str, object], feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        np.savez_compressed(
            save_dir / f"{split_name}.npz",
            X=dataset[f"X_{split_name}"],
            y_binary=dataset[f"y_binary_{split_name}"],
            y_multi=dataset[f"y_multi_{split_name}"],
        )
        meta = dataset[f"meta_{split_name}"]
        if isinstance(meta, pd.DataFrame):
            meta.to_csv(save_dir / f"{split_name}_metadata.csv", index=False)
    with (save_dir / "scaler_v2.pkl").open("wb") as handle:
        pickle.dump(dataset["scaler"], handle)
    (save_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def prepare_ae_data_v2(
    dataset_dir: str | Path = DEFAULT_INSDN_DIR,
    save_dir: str | Path = DEFAULT_AE_PREPARED_DIR,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_train_normal_rows: int | None = DEFAULT_MAX_TRAIN_NORMAL_ROWS,
    split_gap: int = DEFAULT_SPLIT_GAP,
    feature_scheme: str = DEFAULT_FEATURE_SCHEME,
) -> dict[str, object]:
    dataset_dir = resolve_project_path(dataset_dir)
    save_dir = resolve_project_path(save_dir)
    print(f"Loading InSDN rows for LSTM-AE from {dataset_dir} ...")
    flow_df = load_insdn_dataframe(dataset_dir)
    if flow_df.empty:
        raise ValueError("InSDN dataset is empty")
    split = split_grouped_sequences_pre_window(
        flow_df=flow_df,
        seq_len=seq_len,
        stride=stride,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_gap=split_gap,
        feature_scheme=feature_scheme,
    )
    split = _filter_train_normals(split)
    split = _cap_train_normals(split, max_train_normal_rows)
    split = fit_and_transform_v2(split, feature_scheme=feature_scheme)
    config = {
        "dataset_dir": str(dataset_dir),
        "prepared_dir": str(save_dir),
        "feature_scheme": feature_scheme,
        "feature_names": feature_names_for_scheme(feature_scheme),
        "seq_len": int(seq_len),
        "stride": int(stride),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_strategy": "chronological_per_service_key_pre_window_gap",
        "split_gap": int(split_gap),
        "delta_mode": "runtime_rules_only_not_trained_from_csv",
        "source_format": "InSDN_CICFlow_export_mapped_to_ml_core_features",
        "task_type": "lstm_autoencoder",
        "train_normal_only": True,
        "max_train_normal_rows": max_train_normal_rows,
        "normalization": "QuantileTransformer(uniform)+StandardScaler",
        "seed": RANDOM_SEED,
    }
    print(f"Saving AE prepared dataset to {save_dir} ...")
    save_prepared_v2(split, save_dir, config, feature_scheme=feature_scheme)
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare InSDN dataset for LSTM autoencoder training.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_AE_PREPARED_DIR))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-normal-rows", type=int, default=DEFAULT_MAX_TRAIN_NORMAL_ROWS)
    parser.add_argument("--split-gap", type=int, default=DEFAULT_SPLIT_GAP)
    parser.add_argument("--feature-scheme", default=DEFAULT_FEATURE_SCHEME, choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    args = parser.parse_args()
    prepare_ae_data_v2(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_train_normal_rows=args.max_train_normal_rows,
        split_gap=args.split_gap,
        feature_scheme=args.feature_scheme,
    )
