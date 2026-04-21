#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_NORMAL_LABELS = {"normal", "benign", "background"}


def _normalize_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _is_normal_label(value: object, normal_labels: set[str]) -> bool:
    return _normalize_label(value) in normal_labels


def calibrate_threshold(scores: pd.Series, labels: pd.Series, target_fpr: float, score_direction: str, normal_labels: set[str]) -> float:
    valid = pd.notna(scores)
    scores = scores[valid].astype(float)
    labels = labels[valid]
    normal_scores = scores[labels.apply(lambda v: _is_normal_label(v, normal_labels))].to_numpy(dtype=float)
    if normal_scores.size == 0:
        raise ValueError(
            "No usable normal rows found in scores CSV. "
            "If this came from a live runtime run, first prepare a clean benign subset, "
            "or use replay with labels."
        )
    q = min(max(1.0 - float(target_fpr), 0.0), 1.0)
    if score_direction == "lower_is_attack":
        q = 1.0 - q
    return float(np.quantile(normal_scores, q))


def summarize(scores: pd.Series, labels: pd.Series, threshold: float, score_direction: str, normal_labels: set[str]) -> dict[str, float]:
    valid = pd.notna(scores)
    scores_np = scores[valid].astype(float).to_numpy()
    labels = labels[valid]
    is_attack = ~labels.apply(lambda v: _is_normal_label(v, normal_labels)).to_numpy()
    if score_direction == "lower_is_attack":
        pred = scores_np <= threshold
    else:
        pred = scores_np >= threshold
    normal_mask = ~is_attack
    attack_mask = is_attack
    return {
        "n_rows": int(len(scores_np)),
        "n_normal": int(normal_mask.sum()),
        "n_attack": int(attack_mask.sum()),
        "normal_mean": float(scores_np[normal_mask].mean()) if normal_mask.any() else float("nan"),
        "attack_mean": float(scores_np[attack_mask].mean()) if attack_mask.any() else float("nan"),
        "estimated_fpr": float(pred[normal_mask].mean()) if normal_mask.any() else float("nan"),
        "estimated_tpr": float(pred[attack_mask].mean()) if attack_mask.any() else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate runtime threshold from scores.csv.")
    parser.add_argument("--scores-csv", required=True, help="Path to runtime scores.csv produced by TelemetryRuntime")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--score-direction", default="higher_is_attack", choices=["higher_is_attack", "lower_is_attack"])
    parser.add_argument("--bundle-path", default="", help="Optional runtime_bundle.json to update")
    parser.add_argument("--threshold-key", default="runtime_replay_threshold")
    parser.add_argument("--set-runtime-key", action="store_true", help="Also set runtime_threshold_key to --threshold-key")
    parser.add_argument("--normal-labels", nargs="*", default=sorted(DEFAULT_NORMAL_LABELS), help="Labels to treat as normal/benign")
    args = parser.parse_args()

    normal_labels = {_normalize_label(v) for v in args.normal_labels if _normalize_label(v)}
    scores_path = Path(args.scores_csv)
    df = pd.read_csv(scores_path, low_memory=False)
    if "score" not in df.columns:
        raise ValueError("scores.csv must contain a 'score' column")
    if "label" not in df.columns:
        raise ValueError(
            "scores.csv must contain a 'label' column. For unlabeled live runs, first build a clean benign subset."
        )
    if df["label"].isna().all() or (df["label"].astype(str).str.strip() == "").all():
        raise ValueError(
            "scores.csv has no usable labels. This looks like a live runtime log, not a labeled replay log. "
            "Use scripts/prepare_clean_runtime_scores.py on a benign-only run, or replay labeled telemetry."
        )

    threshold = calibrate_threshold(df["score"], df["label"], args.target_fpr, args.score_direction, normal_labels)
    summary = summarize(df["score"], df["label"], threshold, args.score_direction, normal_labels)
    result = {
        "scores_csv": str(scores_path),
        "target_fpr": float(args.target_fpr),
        "score_direction": args.score_direction,
        "recommended_threshold": float(threshold),
        **summary,
    }

    if args.bundle_path:
        bundle_path = Path(args.bundle_path)
        manifest_path = bundle_path / "runtime_bundle.json" if bundle_path.is_dir() else bundle_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        thresholds = dict(manifest.get("thresholds", {}))
        thresholds[args.threshold_key] = float(threshold)
        manifest["thresholds"] = thresholds
        if args.set_runtime_key:
            manifest["runtime_threshold_key"] = args.threshold_key
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result["updated_bundle"] = str(manifest_path)
        result["threshold_key"] = args.threshold_key
        result["runtime_threshold_key"] = manifest.get("runtime_threshold_key", thresholds.get("runtime_threshold_key", "threshold"))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
