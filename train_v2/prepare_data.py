from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .common import DEFAULT_FEATURE_SCHEME, DEFAULT_INSDN_DIR, DEFAULT_POLL_INTERVAL_S, DEFAULT_PREPARED_DIR, LABEL_TO_INDEX, RANDOM_SEED, apply_transform_array, feature_names_for_scheme, resolve_project_path
from .insdn_loader import load_insdn_dataframe

DEFAULT_SEQ_LEN = 4
DEFAULT_STRIDE = 1
DEFAULT_SPLIT_GAP = 1


def _safe_first(group: pd.DataFrame, col: str, default):
    if col in group.columns and len(group[col]):
        value = group[col].iloc[0]
        if pd.isna(value):
            return default
        return value
    return default


def _window_multiclass(labels: list[str]) -> str:
    attacks = [lab for lab in labels if str(lab) != "Normal"]
    if not attacks:
        return "Normal"
    return pd.Series(attacks).value_counts().index[0]


def _iter_windows(group: pd.DataFrame, seq_len: int, stride: int, feature_scheme: str):
    feature_cols = feature_names_for_scheme(feature_scheme)
    feats = group[feature_cols].to_numpy(dtype=np.float32)
    labels = group["label"].astype(str).tolist()
    is_attack = group["is_attack"].to_numpy(dtype=np.int64)
    for start in range(0, len(group) - seq_len + 1, max(1, stride)):
        end = start + seq_len
        label = _window_multiclass(labels[start:end])
        meta = {
            "source_file": _safe_first(group, "source_file", "unknown"),
            "conversation_key": _safe_first(group, "conversation_key", "unknown"),
            "flow_id": _safe_first(group, "flow_id", ""),
            "label": label,
            "label_index": int(LABEL_TO_INDEX.get(label, 0)),
            "is_attack": int(is_attack[start:end].max()),
            "start_timestamp": group["parsed_timestamp"].iloc[start],
            "end_timestamp": group["parsed_timestamp"].iloc[end - 1],
            "src_ip": _safe_first(group, "src_ip", ""),
            "dst_ip": _safe_first(group, "dst_ip", ""),
            "src_port": int(_safe_first(group, "src_port", -1)),
            "dst_port": int(_safe_first(group, "dst_port", -1)),
            "protocol": int(_safe_first(group, "protocol", -1)),
        }
        yield feats[start:end], meta["is_attack"], meta["label_index"], meta


def split_grouped_sequences_pre_window(flow_df: pd.DataFrame, seq_len: int, stride: int, train_ratio: float = 0.7, val_ratio: float = 0.15, split_gap: int = DEFAULT_SPLIT_GAP, feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> dict[str, object]:
    feature_cols = feature_names_for_scheme(feature_scheme)
    buckets = {
        "train_X": [], "train_y_binary": [], "train_y_multi": [], "train_meta": [],
        "val_X": [], "val_y_binary": [], "val_y_multi": [], "val_meta": [],
        "test_X": [], "test_y_binary": [], "test_y_multi": [], "test_meta": [],
    }
    gap = max(int(split_gap), 0)
    for (_, _), group in flow_df.groupby(["source_file", "conversation_key"], sort=False):
        group = group.sort_values(["parsed_timestamp", "row_id"]).reset_index(drop=True)
        if len(group) < seq_len:
            continue
        n = len(group)
        raw_train_end = min(max(1, int(n * train_ratio)), n)
        raw_val_end = min(max(raw_train_end + 1, int(n * (train_ratio + val_ratio))), n)
        train_seg = group.iloc[:raw_train_end].reset_index(drop=True)
        val_start = min(raw_train_end + gap, n)
        val_seg = group.iloc[val_start:raw_val_end].reset_index(drop=True)
        test_start = min(raw_val_end + gap, n)
        test_seg = group.iloc[test_start:].reset_index(drop=True)
        for split_name, seg in (("train", train_seg), ("val", val_seg), ("test", test_seg)):
            if len(seg) < seq_len:
                continue
            for X_win, y_bin, y_multi, meta in _iter_windows(seg, seq_len, stride, feature_scheme):
                buckets[f"{split_name}_X"].append(X_win)
                buckets[f"{split_name}_y_binary"].append(y_bin)
                buckets[f"{split_name}_y_multi"].append(y_multi)
                buckets[f"{split_name}_meta"].append(meta)
    out = {}
    for split_name in ("train", "val", "test"):
        X_list = buckets[f"{split_name}_X"]
        if X_list:
            out[f"X_{split_name}"] = np.stack(X_list).astype(np.float32)
            out[f"y_binary_{split_name}"] = np.asarray(buckets[f"{split_name}_y_binary"], dtype=np.int64)
            out[f"y_multi_{split_name}"] = np.asarray(buckets[f"{split_name}_y_multi"], dtype=np.int64)
            meta_df = pd.DataFrame(buckets[f"{split_name}_meta"])
            meta_df["split"] = split_name
            out[f"meta_{split_name}"] = meta_df
        else:
            out[f"X_{split_name}"] = np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
            out[f"y_binary_{split_name}"] = np.zeros((0,), dtype=np.int64)
            out[f"y_multi_{split_name}"] = np.zeros((0,), dtype=np.int64)
            out[f"meta_{split_name}"] = pd.DataFrame()
    return out


def fit_and_transform(split: dict[str, object], feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> dict[str, object]:
    feature_cols = feature_names_for_scheme(feature_scheme)
    for key in ("X_train", "X_val", "X_test"):
        split[key] = apply_transform_array(split[key], feature_names=feature_cols)
    scaler = RobustScaler()
    train_flat = split["X_train"].reshape(-1, len(feature_cols)) if len(split["X_train"]) else np.zeros((0, len(feature_cols)), dtype=np.float32)
    scaler.fit(train_flat if len(train_flat) else np.zeros((1, len(feature_cols)), dtype=np.float32))
    for key in ("X_train", "X_val", "X_test"):
        X_value = np.asarray(split[key], dtype=np.float32)
        split[key] = scaler.transform(X_value.reshape(-1, len(feature_cols))).reshape(X_value.shape).astype(np.float32) if len(X_value) else X_value
    split["scaler"] = scaler
    return split


def save_prepared(dataset: dict[str, object], save_dir: str | Path, config: dict[str, object], feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> None:
    save_path = resolve_project_path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        np.savez_compressed(save_path / f"{split_name}.npz", X=dataset[f"X_{split_name}"], y_binary=dataset[f"y_binary_{split_name}"], y_multi=dataset[f"y_multi_{split_name}"])
        meta = dataset[f"meta_{split_name}"]
        if isinstance(meta, pd.DataFrame):
            meta.to_csv(save_path / f"{split_name}_metadata.csv", index=False)
    with (save_path / "scaler.pkl").open("wb") as handle:
        pickle.dump(dataset["scaler"], handle)
    scaler = dataset["scaler"]
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    scaler_stats = {"center": np.asarray(scaler.center_, dtype=np.float64).tolist(), "scale": scale.tolist(), "feature_names": feature_names_for_scheme(feature_scheme), "feature_scheme": feature_scheme}
    (save_path / "scaler_stats.json").write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")
    (save_path / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def load_prepared(prepared_dir: str | Path) -> dict[str, object]:
    prepared_path = resolve_project_path(prepared_dir)
    dataset = {}
    for split_name in ("train", "val", "test"):
        payload = np.load(prepared_path / f"{split_name}.npz")
        dataset[f"X_{split_name}"] = payload["X"].astype(np.float32)
        dataset[f"y_binary_{split_name}"] = payload["y_binary"].astype(np.int64)
        dataset[f"y_multi_{split_name}"] = payload["y_multi"].astype(np.int64)
        meta_path = prepared_path / f"{split_name}_metadata.csv"
        dataset[f"meta_{split_name}"] = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    with (prepared_path / "scaler.pkl").open("rb") as handle:
        dataset["scaler"] = pickle.load(handle)
    with (prepared_path / "config.json").open("r", encoding="utf-8") as handle:
        dataset["config"] = json.load(handle)
    return dataset


def prepare_data(dataset_dir: str | Path = DEFAULT_INSDN_DIR, converted_dir: str | Path | None = None, save_dir: str | Path = DEFAULT_PREPARED_DIR, seq_len: int = DEFAULT_SEQ_LEN, stride: int = DEFAULT_STRIDE, train_ratio: float = 0.7, val_ratio: float = 0.15, poll_interval_s: float = DEFAULT_POLL_INTERVAL_S, chunksize: int = 200_000, force_convert: bool = False, split_gap: int = DEFAULT_SPLIT_GAP, feature_scheme: str = DEFAULT_FEATURE_SCHEME) -> dict[str, object]:
    del converted_dir, poll_interval_s, chunksize, force_convert
    dataset_dir = resolve_project_path(dataset_dir)
    save_dir = resolve_project_path(save_dir)
    print(f"Loading InSDN rows from {dataset_dir} ...")
    flow_df = load_insdn_dataframe(dataset_dir)
    if flow_df.empty:
        raise ValueError("InSDN dataset is empty")
    print("Splitting grouped time series first, then creating sequence windows ...")
    split = split_grouped_sequences_pre_window(flow_df, seq_len, stride, train_ratio, val_ratio, split_gap, feature_scheme)
    if len(split["X_train"]) == 0:
        raise ValueError("Prepared train split is empty")
    if len(split["X_val"]) == 0 or len(split["X_test"]) == 0:
        raise ValueError("Validation or test split is empty; reduce seq_len or split_gap.")
    print("Applying signed-log transform + RobustScaler (fit on train only) ...")
    split = fit_and_transform(split, feature_scheme)
    config = {"dataset_dir": str(dataset_dir), "prepared_dir": str(save_dir), "feature_scheme": feature_scheme, "feature_names": feature_names_for_scheme(feature_scheme), "seq_len": seq_len, "stride": stride, "train_ratio": train_ratio, "val_ratio": val_ratio, "split_strategy": "chronological_per_service_key_pre_window_gap", "split_gap": int(split_gap), "delta_mode": "runtime_rules_only_not_trained_from_csv", "source_format": "InSDN_CICFlow_export_mapped_to_openflow_ml_core", "seed": RANDOM_SEED}
    split["config"] = config
    print(f"Saving to {save_dir} ...")
    save_prepared(split, save_dir, config, feature_scheme)
    return split


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sequence data from bundled InSDN CSV files using the ML-core OpenFlow-safe feature contract.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-gap", type=int, default=DEFAULT_SPLIT_GAP)
    parser.add_argument("--feature-scheme", default=DEFAULT_FEATURE_SCHEME, choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    args = parser.parse_args()
    prepare_data(dataset_dir=args.dataset_dir, save_dir=args.save_dir, seq_len=args.seq_len, stride=args.stride, train_ratio=args.train_ratio, val_ratio=args.val_ratio, split_gap=args.split_gap, feature_scheme=args.feature_scheme)
