#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a runtime bundle threshold from live benign scores using a quantile.")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--threshold-key", default="live_benign_threshold")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--runtime-threshold-key", default="")
    args = parser.parse_args()

    scores_csv = Path(args.scores_csv)
    bundle_path = Path(args.bundle_path)
    df = pd.read_csv(scores_csv, low_memory=False)
    if args.score_column not in df.columns:
        raise ValueError(f"Missing score column {args.score_column!r} in {scores_csv}")

    clean = df.copy()
    if "status" in clean.columns:
        clean = clean[clean["status"].astype(str).str.upper().eq("NORMAL")]
    if "support_sources" in clean.columns:
        clean = clean[clean["support_sources"].fillna("").astype(str).str.strip().eq("")]
    if clean.empty:
        raise ValueError("No clean NORMAL rows remain after filtering live scores")

    threshold = float(np.quantile(clean[args.score_column].astype(float).to_numpy(), args.quantile))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    thresholds = dict(bundle.get("thresholds", {}))
    thresholds[args.threshold_key] = threshold
    bundle["thresholds"] = thresholds
    bundle["runtime_threshold_key"] = args.runtime_threshold_key or args.threshold_key
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({
        "scores_csv": str(scores_csv),
        "n_rows_used": int(len(clean)),
        "quantile": float(args.quantile),
        "recommended_threshold": threshold,
        "bundle_path": str(bundle_path),
        "runtime_threshold_key": bundle["runtime_threshold_key"],
    }, indent=2))


if __name__ == "__main__":
    main()
