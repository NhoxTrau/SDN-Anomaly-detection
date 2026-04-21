#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

try:
    from sdn_nids_realtime.runtime.replay_engine import replay_csv
    from sdn_nids_realtime.runtime.telemetry_runtime import TelemetryRuntime
except ModuleNotFoundError:
    from runtime.replay_engine import replay_csv
    from runtime.telemetry_runtime import TelemetryRuntime

from build_insdn_runtime_replay import build_runtime_replay_dataframe
from calibrate_threshold_from_scores import DEFAULT_NORMAL_LABELS, _normalize_label, calibrate_threshold, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled runtime replay from InSDN, replay it through runtime, then calibrate threshold.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--run-id", default="replay_insdn_calibration")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=6)
    parser.add_argument("--take-last-ratio", type=float, default=0.20)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--threshold-key", default="runtime_replay_threshold")
    parser.add_argument("--set-runtime-key", action="store_true")
    parser.add_argument("--normal-only", action="store_true", help="Calibrate from benign-only replay rows.")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    run_dir = runtime_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_csv_path = run_dir / ("replay_input_normal.csv" if args.normal_only else "replay_input_labeled.csv")

    replay_df, build_summary = build_runtime_replay_dataframe(
        dataset_dir=args.dataset_dir,
        poll_interval_s=args.poll_interval,
        max_polls=args.max_polls,
        take_last_ratio=args.take_last_ratio,
        normal_only=args.normal_only,
    )
    replay_df.to_csv(replay_csv_path, index=False)

    # clear stale outputs from an earlier run with the same run_id
    for stale_name in ("scores.csv", "alerts.csv", "telemetry_stream.csv", "dashboard_state.json"):
        stale_path = run_dir / stale_name
        if stale_path.exists():
            try:
                stale_path.unlink()
            except Exception:
                pass

    runtime = TelemetryRuntime(runtime_root=runtime_root, bundle_path=args.bundle_path, run_id=args.run_id)
    replay_csv(runtime, replay_csv_path, feature_scheme=runtime.feature_scheme, speed=args.speed)
    runtime._flush_state(force=True)
    runtime.wait_for_csv_drain(timeout_s=max(30.0, min(180.0, len(replay_df) / 2000.0)))

    scores_csv = run_dir / "scores.csv"
    import pandas as pd
    deadline = time.time() + max(15.0, min(120.0, len(replay_df) / 1000.0))
    while time.time() <= deadline:
        if scores_csv.exists() and scores_csv.stat().st_size > 0:
            break
        time.sleep(0.1)
    if not scores_csv.exists() or scores_csv.stat().st_size <= 0:
        telemetry_csv = run_dir / "telemetry_stream.csv"
        detail = []
        if telemetry_csv.exists():
            detail.append(f"telemetry_stream.csv exists ({telemetry_csv.stat().st_size} bytes)")
        if (run_dir / "dashboard_state.json").exists():
            detail.append("dashboard_state.json exists")
        raise FileNotFoundError(
            f"Replay finished but scores.csv was not created at {scores_csv}. "
            f"This usually means runtime rows never reached seq_len={runtime.seq_len}, "
            f"or CSV writer had not drained yet. {'; '.join(detail)}"
        )
    scores_df = pd.read_csv(scores_csv, low_memory=False)
    if scores_df.empty:
        raise ValueError(f"scores.csv is empty at {scores_csv}; replay produced no scored rows")
    normal_labels = {_normalize_label(v) for v in DEFAULT_NORMAL_LABELS}
    threshold = calibrate_threshold(scores_df["score"], scores_df["label"], args.target_fpr, "higher_is_attack", normal_labels)
    summary = summarize(scores_df["score"], scores_df["label"], threshold, "higher_is_attack", normal_labels)

    manifest_path = Path(args.bundle_path)
    manifest_path = manifest_path / "runtime_bundle.json" if manifest_path.is_dir() else manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thresholds = dict(manifest.get("thresholds", {}))
    thresholds[args.threshold_key] = float(threshold)
    manifest["thresholds"] = thresholds
    if args.set_runtime_key:
        manifest["runtime_threshold_key"] = args.threshold_key
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = {
        **build_summary,
        "replay_csv": str(replay_csv_path),
        "scores_csv": str(scores_csv),
        "target_fpr": float(args.target_fpr),
        "recommended_threshold": float(threshold),
        **summary,
        "updated_bundle": str(manifest_path),
        "threshold_key": args.threshold_key,
        "runtime_threshold_key": manifest.get("runtime_threshold_key", args.threshold_key),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
