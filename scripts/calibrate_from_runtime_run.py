#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_threshold_from_scores import calibrate_threshold, summarize, _normalize_label, DEFAULT_NORMAL_LABELS
from prepare_clean_runtime_scores import prepare_clean_subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clean benign subset from a runtime run and calibrate threshold.")
    parser.add_argument("--run-dir", required=True, help="Path like runtime_logs/realtime_benign")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--score-direction", default="higher_is_attack", choices=["higher_is_attack", "lower_is_attack"])
    parser.add_argument("--bundle-path", default="")
    parser.add_argument("--threshold-key", default="runtime_benign_threshold")
    parser.add_argument("--set-runtime-key", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores_csv = run_dir / "scores.csv"
    alerts_csv = run_dir / "alerts.csv"
    out_df, prep_summary = prepare_clean_subset(scores_csv, alerts_csv if alerts_csv.exists() else None)
    clean_csv = run_dir / "scores_clean_benign.csv"
    out_df.to_csv(clean_csv, index=False)

    normal_labels = {_normalize_label(v) for v in DEFAULT_NORMAL_LABELS}
    threshold = calibrate_threshold(out_df["score"], out_df["label"], args.target_fpr, args.score_direction, normal_labels)
    summary = summarize(out_df["score"], out_df["label"], threshold, args.score_direction, normal_labels)
    result = {
        **prep_summary,
        "clean_scores_csv": str(clean_csv),
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
