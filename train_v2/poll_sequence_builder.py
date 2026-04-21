from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_FEATURE_SCHEME,
    DEFAULT_INSDN_DIR,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_PREPARED_DIR,
    LABEL_TO_INDEX,
    RANDOM_SEED,
    feature_names_for_scheme,
    normalise_label,
    resolve_project_path,
)
from .insdn_loader import load_insdn_dataframe
from .prepare_data import (
    DEFAULT_SEQ_LEN,
    DEFAULT_SPLIT_GAP,
    DEFAULT_STRIDE,
    fit_and_transform,
    save_prepared,
)

DEFAULT_MAX_POLLS = 6
DEFAULT_KEY_MODE = "flow"
DEFAULT_WINDOW_LABEL_MODE = "any_attack_priority"
DEFAULT_TEST_HOLDOUT_MODE = "temporal"
DEFAULT_POLL_PREPARED_DIR = DEFAULT_PREPARED_DIR.parent / "prepared_poll_classifier"
DEFAULT_POLL_SPLIT_GAP = 0
DEFAULT_MIN_VAL_GROUPS_PER_LABEL = 1
DEFAULT_MIN_TEST_GROUPS_PER_LABEL = 1


def _bucket_anchor_from_timestamps(ts_seconds: pd.Series, poll_interval_s: float) -> float:
    start_ts = float(ts_seconds.min()) if len(ts_seconds) else 0.0
    return math.floor(start_ts / poll_interval_s) * poll_interval_s


def _service_key(frame: pd.DataFrame) -> pd.Series:
    src_ip = frame["src_ip"].astype(str).str.strip()
    dst_ip = frame["dst_ip"].astype(str).str.strip()
    dst_port = pd.to_numeric(frame["dst_port"], errors="coerce").fillna(-1).astype(int).astype(str)
    protocol = pd.to_numeric(frame["protocol"], errors="coerce").fillna(-1).astype(int).astype(str)
    return src_ip + "__" + dst_ip + "__" + dst_port + "__" + protocol


def _flow_key(frame: pd.DataFrame) -> pd.Series:
    src_ip = frame["src_ip"].astype(str).str.strip()
    dst_ip = frame["dst_ip"].astype(str).str.strip()
    src_port = pd.to_numeric(frame["src_port"], errors="coerce").fillna(-1).astype(int).astype(str)
    dst_port = pd.to_numeric(frame["dst_port"], errors="coerce").fillna(-1).astype(int).astype(str)
    protocol = pd.to_numeric(frame["protocol"], errors="coerce").fillna(-1).astype(int).astype(str)
    return src_ip + "__" + dst_ip + "__" + src_port + "__" + dst_port + "__" + protocol


def _assign_conversation_key(df: pd.DataFrame, key_mode: str) -> pd.DataFrame:
    out = df.copy()
    if key_mode == "flow":
        out["conversation_key"] = _flow_key(out)
    elif key_mode == "service":
        out["conversation_key"] = _service_key(out)
    else:
        raise ValueError(f"Unsupported key_mode={key_mode!r}. Use 'service' or 'flow'.")
    return out


def _safe_first(group: pd.DataFrame, col: str, default):
    if col in group.columns and len(group[col]):
        value = group[col].iloc[0]
        if pd.isna(value):
            return default
        return value
    return default


def _label_priority(labels: Iterable[str]) -> str:
    normalised = [normalise_label(v) for v in labels]
    attacks = [lab for lab in normalised if lab != "Normal"]
    if not attacks:
        return "Normal"
    counts = Counter(attacks)
    return max(sorted(counts), key=lambda lab: (counts[lab], lab))


def _window_multiclass(labels: list[str], mode: str = DEFAULT_WINDOW_LABEL_MODE) -> str:
    labels = [normalise_label(v) for v in labels]
    if not labels:
        return "Normal"
    if mode == "majority":
        counts = Counter(labels)
        return max(sorted(counts), key=lambda lab: (counts[lab], lab))
    if mode == "last_step":
        for label in reversed(labels):
            if label != "Normal":
                return label
        return labels[-1]
    if mode == "any_attack_priority":
        attacks = [lab for lab in labels if lab != "Normal"]
        if not attacks:
            return "Normal"
        counts = Counter(attacks)
        return max(sorted(counts), key=lambda lab: (counts[lab], lab))
    raise ValueError(f"Unsupported window_label_mode={mode!r}")


def _make_snapshot_rows(
    flow_df: pd.DataFrame,
    poll_interval_s: float,
    max_polls: int,
    key_mode: str,
) -> pd.DataFrame:
    if flow_df.empty:
        return pd.DataFrame()

    base = _assign_conversation_key(flow_df, key_mode=key_mode)
    base = base.sort_values(["source_file", "parsed_timestamp", "row_id"]).reset_index(drop=True)

    base["start_ts"] = base["parsed_timestamp"].astype("int64") / 1e9
    base["duration_s"] = pd.to_numeric(base["flow_duration_s"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # Preserve real duration so short bursts are not stretched to a full poll interval.
    duration_safe = np.maximum(base["duration_s"].to_numpy(dtype=np.float64), 1e-6)
    base["effective_duration_s"] = duration_safe
    base["end_ts"] = base["start_ts"] + duration_safe

    anchor = _bucket_anchor_from_timestamps(base["start_ts"], poll_interval_s)
    rows: list[dict[str, object]] = []

    for record in base.itertuples(index=False):
        start_ts = float(record.start_ts)
        duration_s = float(record.effective_duration_s)
        end_ts = float(record.end_ts)
        clipped_end_ts = min(end_ts, start_ts + float(max_polls) * float(poll_interval_s))

        first_poll_idx = int(math.ceil((start_ts - anchor) / poll_interval_s))
        last_poll_idx = int(math.floor((clipped_end_ts - anchor) / poll_interval_s + 1e-9))
        if last_poll_idx < first_poll_idx:
            last_poll_idx = first_poll_idx

        for poll_idx in range(first_poll_idx, last_poll_idx + 1):
            poll_ts = anchor + poll_idx * poll_interval_s
            elapsed = min(max(poll_ts - start_ts, 0.0), duration_s)
            frac = min(max(elapsed / max(duration_s, 1e-6), 0.0), 1.0)
            packet_count = float(record.packet_count) * frac
            byte_count = float(record.byte_count) * frac
            flow_duration_s = elapsed
            packet_rate = packet_count / max(flow_duration_s, 1e-6)
            byte_rate = byte_count / max(flow_duration_s, 1e-6)
            avg_packet_size = byte_count / max(packet_count, 1.0)
            rows.append(
                {
                    "source_file": str(record.source_file),
                    "conversation_key": str(record.conversation_key),
                    "flow_id": str(record.flow_id),
                    "poll_index": int(poll_idx),
                    "parsed_timestamp": pd.to_datetime(poll_ts, unit="s", utc=True),
                    "src_ip": str(record.src_ip),
                    "dst_ip": str(record.dst_ip),
                    "src_port": int(record.src_port),
                    "dst_port": int(record.dst_port),
                    "protocol": int(record.protocol),
                    "label": normalise_label(record.label),
                    "label_index": int(record.label_index),
                    "is_attack": int(record.is_attack),
                    "packet_count": packet_count,
                    "byte_count": byte_count,
                    "flow_duration_s": flow_duration_s,
                    "packet_rate": packet_rate,
                    "byte_rate": byte_rate,
                    "avg_packet_size": avg_packet_size,
                    "dst_port_norm": float(record.dst_port_norm),
                    "protocol_tcp": float(record.protocol_tcp),
                    "protocol_udp": float(record.protocol_udp),
                    "protocol_icmp": float(record.protocol_icmp),
                    "subflow_count_same_key": 1.0,
                    "unique_src_ports_same_key": 1.0,
                    "poll_interval_s": float(poll_interval_s),
                    "snapshot_frac": frac,
                    "row_id": int(record.row_id),
                }
            )

    poll_df = pd.DataFrame(rows)
    if poll_df.empty:
        return poll_df

    def _agg_frame(group: pd.DataFrame) -> pd.Series:
        label = _label_priority(group["label"].tolist())
        return pd.Series(
            {
                "source_file": str(group["source_file"].iloc[0]),
                "conversation_key": str(group["conversation_key"].iloc[0]),
                "flow_id": str(group["flow_id"].iloc[0]),
                "parsed_timestamp": group["parsed_timestamp"].iloc[0],
                "src_ip": str(group["src_ip"].iloc[0]),
                "dst_ip": str(group["dst_ip"].iloc[0]),
                "src_port": int(group["src_port"].iloc[0]) if len(group["src_port"].unique()) == 1 else -1,
                "dst_port": int(group["dst_port"].iloc[0]),
                "protocol": int(group["protocol"].iloc[0]),
                "label": label,
                "label_index": int(group.loc[group["label"] == label, "label_index"].iloc[0]) if label in set(group["label"]) else 0,
                "is_attack": int((group["is_attack"] > 0).any()),
                "packet_count": float(group["packet_count"].sum()),
                "byte_count": float(group["byte_count"].sum()),
                "flow_duration_s": float(group["flow_duration_s"].max()),
                "packet_rate": float(group["packet_count"].sum() / max(group["flow_duration_s"].max(), 1e-6)),
                "byte_rate": float(group["byte_count"].sum() / max(group["flow_duration_s"].max(), 1e-6)),
                "avg_packet_size": float(group["byte_count"].sum() / max(group["packet_count"].sum(), 1.0)),
                "dst_port_norm": float(group["dst_port_norm"].iloc[0]),
                "protocol_tcp": float(group["protocol_tcp"].iloc[0]),
                "protocol_udp": float(group["protocol_udp"].iloc[0]),
                "protocol_icmp": float(group["protocol_icmp"].iloc[0]),
                "subflow_count_same_key": float(len(group)),
                "unique_src_ports_same_key": float(group["src_port"].astype(int).replace(-1, np.nan).dropna().nunique()),
                "poll_interval_s": float(group["poll_interval_s"].iloc[0]),
                "snapshot_frac": float(group["snapshot_frac"].max()),
                "poll_index": int(group["poll_index"].iloc[0]),
                "row_id": int(group["poll_index"].iloc[0]),
            }
        )

    agg_rows = []
    for _, group in poll_df.groupby(["source_file", "conversation_key", "poll_index"], sort=True):
        agg_rows.append(_agg_frame(group).to_dict())
    agg = pd.DataFrame(agg_rows)
    agg = agg.sort_values(["source_file", "conversation_key", "parsed_timestamp", "poll_index"]).reset_index(drop=True)
    return agg


def build_insdn_poll_dataframe(
    dataset_dir: str | Path = DEFAULT_INSDN_DIR,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    max_polls: int = DEFAULT_MAX_POLLS,
    key_mode: str = DEFAULT_KEY_MODE,
) -> pd.DataFrame:
    dataset_dir = resolve_project_path(dataset_dir)
    flow_df = load_insdn_dataframe(dataset_dir)
    if flow_df.empty:
        raise ValueError("InSDN dataset is empty")
    return _make_snapshot_rows(flow_df, poll_interval_s=float(poll_interval_s), max_polls=int(max_polls), key_mode=key_mode)


def _iter_windows(group: pd.DataFrame, seq_len: int, stride: int, feature_scheme: str, window_label_mode: str):
    feature_cols = feature_names_for_scheme(feature_scheme)
    feats = group[feature_cols].to_numpy(dtype=np.float32)
    labels = group["label"].astype(str).tolist()
    is_attack = group["is_attack"].to_numpy(dtype=np.int64)
    for start in range(0, len(group) - seq_len + 1, max(1, stride)):
        end = start + seq_len
        label = _window_multiclass(labels[start:end], mode=window_label_mode)
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


def _group_windows_for_split(flow_df: pd.DataFrame, seq_len: int, stride: int, feature_scheme: str, window_label_mode: str):
    groups: list[dict[str, object]] = []
    for (source_file, conversation_key), group in flow_df.groupby(["source_file", "conversation_key"], sort=False):
        group = group.sort_values(["parsed_timestamp", "row_id"]).reset_index(drop=True)
        if len(group) < seq_len:
            continue
        windows = list(_iter_windows(group, seq_len, stride, feature_scheme, window_label_mode))
        if not windows:
            continue
        labels = [w[3]["label"] for w in windows]
        group_label = _window_multiclass(labels, mode="any_attack_priority")
        groups.append(
            {
                "source_file": str(source_file),
                "conversation_key": str(conversation_key),
                "label": group_label,
                "windows": windows,
                "n_windows": len(windows),
                "start_timestamp": pd.Timestamp(group["parsed_timestamp"].iloc[0]),
                "end_timestamp": pd.Timestamp(group["parsed_timestamp"].iloc[-1]),
            }
        )
    return groups


def _counts_for_label(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n <= 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1
    n_test = max(1, int(round(n * max(0.0, 1.0 - train_ratio - val_ratio))))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    while n_train < 1 and n_val > 1:
        n_val -= 1
        n_train += 1
    while n_train < 1 and n_test > 1:
        n_test -= 1
        n_train += 1
    if n_train < 1:
        n_train = 1
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
    return n_train, n_val, n_test


def _allocate_groups_hybrid(
    groups: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_holdout_mode: str,
) -> dict[str, list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in groups:
        by_label[str(item["label"])].append(item)

    split_groups: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    for label, items in by_label.items():
        items = sorted(items, key=lambda g: (g["start_timestamp"], g["end_timestamp"], g["conversation_key"]))
        n_train, n_val, n_test = _counts_for_label(len(items), train_ratio, val_ratio)

        if test_holdout_mode == "temporal":
            test_items = items[-n_test:] if n_test > 0 else []
            remaining = items[:-n_test] if n_test > 0 else items
        else:
            order = rng.permutation(len(items))
            shuffled = [items[i] for i in order]
            test_items = shuffled[:n_test]
            remaining = shuffled[n_test:]

        remaining = list(remaining)
        if len(remaining) > 1:
            order = rng.permutation(len(remaining))
            remaining = [remaining[i] for i in order]
        val_items = remaining[:n_val]
        train_items = remaining[n_val:]
        if not train_items and val_items:
            train_items.append(val_items.pop())

        split_groups["train"].extend(train_items)
        split_groups["val"].extend(val_items)
        split_groups["test"].extend(test_items)
    return split_groups


def _materialize_split(
    allocated: dict[str, list[dict[str, object]]],
    seq_len: int,
    feature_scheme: str,
) -> dict[str, object]:
    feature_cols = feature_names_for_scheme(feature_scheme)
    out: dict[str, object] = {}
    for split_name in ("train", "val", "test"):
        X_list: list[np.ndarray] = []
        y_binary: list[int] = []
        y_multi: list[int] = []
        meta_list: list[dict[str, object]] = []
        for group in allocated[split_name]:
            for X_win, y_bin, y_mul, meta in group["windows"]:
                X_list.append(X_win)
                y_binary.append(y_bin)
                y_multi.append(y_mul)
                meta_list.append(meta)
        if X_list:
            out[f"X_{split_name}"] = np.stack(X_list).astype(np.float32)
            out[f"y_binary_{split_name}"] = np.asarray(y_binary, dtype=np.int64)
            out[f"y_multi_{split_name}"] = np.asarray(y_multi, dtype=np.int64)
            meta_df = pd.DataFrame(meta_list)
            meta_df["split"] = split_name
            out[f"meta_{split_name}"] = meta_df
        else:
            out[f"X_{split_name}"] = np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32)
            out[f"y_binary_{split_name}"] = np.zeros((0,), dtype=np.int64)
            out[f"y_multi_{split_name}"] = np.zeros((0,), dtype=np.int64)
            out[f"meta_{split_name}"] = pd.DataFrame()
    return out


def _build_split_report(allocated: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    report: dict[str, object] = {"splits": {}}
    for split_name in ("train", "val", "test"):
        label_groups: Counter[str] = Counter()
        label_windows: Counter[str] = Counter()
        source_files: Counter[str] = Counter()
        for item in allocated[split_name]:
            label = str(item["label"])
            label_groups[label] += 1
            label_windows[label] += int(item["n_windows"])
            source_files[str(item["source_file"])] += 1
        report["splits"][split_name] = {
            "n_groups": int(sum(label_groups.values())),
            "n_windows": int(sum(label_windows.values())),
            "groups_per_label": dict(sorted(label_groups.items())),
            "windows_per_label": dict(sorted(label_windows.items())),
            "source_files": dict(sorted(source_files.items())),
        }
    return report


def _validate_split_label_coverage(
    report: dict[str, object],
    min_val_groups_per_label: int,
    min_test_groups_per_label: int,
) -> list[str]:
    warnings: list[str] = []
    train_labels = set(report["splits"].get("train", {}).get("groups_per_label", {}).keys())
    for label in sorted(train_labels):
        if label == "Normal":
            continue
        val_groups = int(report["splits"].get("val", {}).get("groups_per_label", {}).get(label, 0))
        test_groups = int(report["splits"].get("test", {}).get("groups_per_label", {}).get(label, 0))
        if val_groups < int(min_val_groups_per_label):
            warnings.append(f"Label {label} has only {val_groups} validation groups (target >= {min_val_groups_per_label}).")
        if test_groups < int(min_test_groups_per_label):
            warnings.append(f"Label {label} has only {test_groups} test groups (target >= {min_test_groups_per_label}).")
    return warnings


def split_grouped_sequences_post_window(
    flow_df: pd.DataFrame,
    seq_len: int,
    stride: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_scheme: str = DEFAULT_FEATURE_SCHEME,
    window_label_mode: str = DEFAULT_WINDOW_LABEL_MODE,
    test_holdout_mode: str = DEFAULT_TEST_HOLDOUT_MODE,
) -> tuple[dict[str, object], dict[str, object]]:
    groups = _group_windows_for_split(flow_df, seq_len, stride, feature_scheme, window_label_mode)
    if not groups:
        raise ValueError("No sequence windows could be built from poll rows")
    allocated = _allocate_groups_hybrid(groups, train_ratio=train_ratio, val_ratio=val_ratio, test_holdout_mode=test_holdout_mode)
    split = _materialize_split(allocated, seq_len=seq_len, feature_scheme=feature_scheme)
    report = _build_split_report(allocated)
    return split, report


def prepare_poll_sequences(
    dataset_dir: str | Path = DEFAULT_INSDN_DIR,
    save_dir: str | Path = DEFAULT_POLL_PREPARED_DIR,
    poll_csv_path: str | Path | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    split_gap: int = DEFAULT_POLL_SPLIT_GAP,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    max_polls: int = DEFAULT_MAX_POLLS,
    feature_scheme: str = DEFAULT_FEATURE_SCHEME,
    key_mode: str = DEFAULT_KEY_MODE,
    window_label_mode: str = DEFAULT_WINDOW_LABEL_MODE,
    test_holdout_mode: str = DEFAULT_TEST_HOLDOUT_MODE,
    min_val_groups_per_label: int = DEFAULT_MIN_VAL_GROUPS_PER_LABEL,
    min_test_groups_per_label: int = DEFAULT_MIN_TEST_GROUPS_PER_LABEL,
) -> dict[str, object]:
    del split_gap  # legacy arg kept for CLI compatibility; split is now group-aware.
    feature_names = feature_names_for_scheme(feature_scheme)
    save_dir = resolve_project_path(save_dir)
    poll_csv_path = resolve_project_path(poll_csv_path) if poll_csv_path else (save_dir / "poll_rows.csv")
    poll_df = build_insdn_poll_dataframe(dataset_dir=dataset_dir, poll_interval_s=poll_interval_s, max_polls=max_polls, key_mode=key_mode)
    if poll_df.empty:
        raise ValueError("Synthetic poll dataframe is empty")
    poll_csv_path.parent.mkdir(parents=True, exist_ok=True)
    poll_df.to_csv(poll_csv_path, index=False)

    split, report = split_grouped_sequences_post_window(
        flow_df=poll_df,
        seq_len=seq_len,
        stride=stride,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        feature_scheme=feature_scheme,
        window_label_mode=window_label_mode,
        test_holdout_mode=test_holdout_mode,
    )
    if len(split["X_train"]) == 0:
        raise ValueError("Prepared train split is empty after poll transformation")
    if len(split["X_val"]) == 0 or len(split["X_test"]) == 0:
        raise ValueError("Validation or test split is empty after poll transformation")

    coverage_warnings = _validate_split_label_coverage(report, min_val_groups_per_label, min_test_groups_per_label)
    split = fit_and_transform(split, feature_scheme=feature_scheme)
    config = {
        "dataset_dir": str(resolve_project_path(dataset_dir)),
        "prepared_dir": str(save_dir),
        "poll_csv_path": str(poll_csv_path),
        "feature_scheme": feature_scheme,
        "feature_names": feature_names,
        "seq_len": int(seq_len),
        "stride": int(stride),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_strategy": f"hybrid_{test_holdout_mode}_test_then_group_split",
        "split_gap": 0,
        "source_format": "InSDN_completed_flows_transformed_to_openflow_poll_snapshots",
        "poll_interval_s": float(poll_interval_s),
        "max_polls_per_flow": int(max_polls),
        "key_mode": key_mode,
        "window_label_mode": window_label_mode,
        "test_holdout_mode": test_holdout_mode,
        "min_val_groups_per_label": int(min_val_groups_per_label),
        "min_test_groups_per_label": int(min_test_groups_per_label),
        "coverage_warnings": coverage_warnings,
        "random_seed": int(RANDOM_SEED),
    }
    split["config"] = config
    save_prepared(split, save_dir=save_dir, config=config, feature_scheme=feature_scheme)
    (save_dir / "split_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if coverage_warnings:
        (save_dir / "split_warnings.json").write_text(json.dumps({"warnings": coverage_warnings}, indent=2), encoding="utf-8")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenFlow-like poll snapshot sequences from InSDN completed-flow rows.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_POLL_PREPARED_DIR))
    parser.add_argument("--poll-csv-path", default="")
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--max-polls", type=int, default=DEFAULT_MAX_POLLS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-gap", type=int, default=DEFAULT_POLL_SPLIT_GAP)
    parser.add_argument("--feature-scheme", default=DEFAULT_FEATURE_SCHEME, choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    parser.add_argument("--key-mode", default=DEFAULT_KEY_MODE, choices=["service", "flow"])
    parser.add_argument("--window-label-mode", default=DEFAULT_WINDOW_LABEL_MODE, choices=["any_attack_priority", "majority", "last_step"])
    parser.add_argument("--test-holdout-mode", default=DEFAULT_TEST_HOLDOUT_MODE, choices=["temporal", "stratified"])
    parser.add_argument("--min-val-groups-per-label", type=int, default=DEFAULT_MIN_VAL_GROUPS_PER_LABEL)
    parser.add_argument("--min-test-groups-per-label", type=int, default=DEFAULT_MIN_TEST_GROUPS_PER_LABEL)
    args = parser.parse_args()
    prepare_poll_sequences(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        poll_csv_path=(args.poll_csv_path or None),
        poll_interval_s=args.poll_interval_s,
        max_polls=args.max_polls,
        seq_len=args.seq_len,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_gap=args.split_gap,
        feature_scheme=args.feature_scheme,
        key_mode=args.key_mode,
        window_label_mode=args.window_label_mode,
        test_holdout_mode=args.test_holdout_mode,
        min_val_groups_per_label=args.min_val_groups_per_label,
        min_test_groups_per_label=args.min_test_groups_per_label,
    )


if __name__ == "__main__":
    main()
