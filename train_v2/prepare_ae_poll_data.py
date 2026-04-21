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
from .poll_sequence_builder import (
    DEFAULT_KEY_MODE,
    DEFAULT_MAX_POLLS,
    DEFAULT_POLL_SPLIT_GAP,
    DEFAULT_TEST_HOLDOUT_MODE,
    DEFAULT_WINDOW_LABEL_MODE,
    prepare_poll_sequences,
)
from .prepare_data import DEFAULT_SEQ_LEN, DEFAULT_STRIDE, load_prepared

DEFAULT_MAX_TRAIN_NORMAL_ROWS = 300_000
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


def save_prepared_v2(dataset: dict[str, object], save_dir: Path, config: dict[str, object]) -> None:
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


def _base_split_from_classifier_prepared(source_prepared_dir: str | Path) -> dict[str, object]:
    source_prepared_dir = resolve_project_path(source_prepared_dir)
    base = load_prepared(source_prepared_dir)
    return {
        "X_train": np.asarray(base["X_train"], dtype=np.float32).copy(),
        "X_val": np.asarray(base["X_val"], dtype=np.float32).copy(),
        "X_test": np.asarray(base["X_test"], dtype=np.float32).copy(),
        "y_binary_train": np.asarray(base["y_binary_train"], dtype=np.int64).copy(),
        "y_binary_val": np.asarray(base["y_binary_val"], dtype=np.int64).copy(),
        "y_binary_test": np.asarray(base["y_binary_test"], dtype=np.int64).copy(),
        "y_multi_train": np.asarray(base["y_multi_train"], dtype=np.int64).copy(),
        "y_multi_val": np.asarray(base["y_multi_val"], dtype=np.int64).copy(),
        "y_multi_test": np.asarray(base["y_multi_test"], dtype=np.int64).copy(),
        "meta_train": base.get("meta_train", pd.DataFrame()).copy(),
        "meta_val": base.get("meta_val", pd.DataFrame()).copy(),
        "meta_test": base.get("meta_test", pd.DataFrame()).copy(),
        "source_config": base.get("config", {}),
    }


def prepare_ae_poll_data(
    dataset_dir: str | Path = DEFAULT_INSDN_DIR,
    save_dir: str | Path = DEFAULT_AE_PREPARED_DIR,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_gap: int = DEFAULT_POLL_SPLIT_GAP,
    feature_scheme: str = DEFAULT_FEATURE_SCHEME,
    poll_interval_s: float = 1.0,
    max_polls: int = DEFAULT_MAX_POLLS,
    key_mode: str = DEFAULT_KEY_MODE,
    max_train_normal_rows: int | None = DEFAULT_MAX_TRAIN_NORMAL_ROWS,
    source_prepared_dir: str | Path | None = None,
    window_label_mode: str = DEFAULT_WINDOW_LABEL_MODE,
    test_holdout_mode: str = DEFAULT_TEST_HOLDOUT_MODE,
) -> dict[str, object]:
    dataset_dir = resolve_project_path(dataset_dir)
    save_dir = resolve_project_path(save_dir)
    if source_prepared_dir:
        split = _base_split_from_classifier_prepared(source_prepared_dir)
        source_config = dict(split.pop("source_config", {}))
    else:
        built = prepare_poll_sequences(
            dataset_dir=dataset_dir,
            save_dir=save_dir.parent / f"_tmp_classifier_shared_{poll_interval_s}" ,
            seq_len=seq_len,
            stride=stride,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_gap=split_gap,
            feature_scheme=feature_scheme,
            poll_interval_s=poll_interval_s,
            max_polls=max_polls,
            key_mode=key_mode,
            window_label_mode=window_label_mode,
            test_holdout_mode=test_holdout_mode,
        )
        source_config = dict(built.get("config", {}))
        split = {k: v for k, v in built.items() if k.startswith(("X_", "y_", "meta_"))}

    split = _filter_train_normals(split)
    split = _cap_train_normals(split, max_train_normal_rows)
    split = fit_and_transform_v2(split, feature_scheme=feature_scheme)
    config = {
        "dataset_dir": str(dataset_dir),
        "prepared_dir": str(save_dir),
        "feature_scheme": feature_scheme,
        "feature_names": feature_names_for_scheme(feature_scheme),
        "seq_len": int(seq_len or source_config.get("seq_len", DEFAULT_SEQ_LEN)),
        "stride": int(stride or source_config.get("stride", DEFAULT_STRIDE)),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_strategy": str(source_config.get("split_strategy", "shared_classifier_split_reused_for_ae")),
        "split_gap": int(source_config.get("split_gap", split_gap)),
        "task_type": "lstm_autoencoder_poll_snapshots",
        "source_format": str(source_config.get("source_format", "InSDN_completed_flows_transformed_to_openflow_poll_snapshots")),
        "poll_interval_s": float(source_config.get("poll_interval_s", poll_interval_s)),
        "max_polls_per_flow": int(source_config.get("max_polls_per_flow", max_polls)),
        "key_mode": str(source_config.get("key_mode", key_mode)),
        "window_label_mode": str(source_config.get("window_label_mode", window_label_mode)),
        "test_holdout_mode": str(source_config.get("test_holdout_mode", test_holdout_mode)),
        "train_normal_only": True,
        "max_train_normal_rows": max_train_normal_rows,
        "normalization": "QuantileTransformer(uniform)+StandardScaler",
        "source_prepared_dir": str(resolve_project_path(source_prepared_dir)) if source_prepared_dir else None,
        "seed": RANDOM_SEED,
    }
    save_prepared_v2(split, save_dir, config)
    split["config"] = config
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare synthetic OpenFlow-poll sequence data for LSTM autoencoder training.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_AE_PREPARED_DIR))
    parser.add_argument("--source-prepared-dir", default="")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-gap", type=int, default=DEFAULT_POLL_SPLIT_GAP)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=DEFAULT_MAX_POLLS)
    parser.add_argument("--key-mode", default=DEFAULT_KEY_MODE, choices=["service", "flow"])
    parser.add_argument("--window-label-mode", default=DEFAULT_WINDOW_LABEL_MODE, choices=["any_attack_priority", "majority", "last_step"])
    parser.add_argument("--test-holdout-mode", default=DEFAULT_TEST_HOLDOUT_MODE, choices=["temporal", "stratified"])
    parser.add_argument("--max-train-normal-rows", type=int, default=DEFAULT_MAX_TRAIN_NORMAL_ROWS)
    parser.add_argument("--feature-scheme", default=DEFAULT_FEATURE_SCHEME, choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    args = parser.parse_args()
    prepare_ae_poll_data(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        source_prepared_dir=(args.source_prepared_dir or None),
        seq_len=args.seq_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_gap=args.split_gap,
        feature_scheme=args.feature_scheme,
        poll_interval_s=args.poll_interval_s,
        max_polls=args.max_polls,
        key_mode=args.key_mode,
        window_label_mode=args.window_label_mode,
        test_holdout_mode=args.test_holdout_mode,
        max_train_normal_rows=args.max_train_normal_rows,
    )


if __name__ == "__main__":
    main()
