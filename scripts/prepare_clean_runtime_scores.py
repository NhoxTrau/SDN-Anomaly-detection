#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def prepare_clean_subset(scores_csv: Path, alerts_csv: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(scores_csv, low_memory=False)
    if "score" not in df.columns or "entity_key" not in df.columns:
        raise ValueError("scores.csv must contain at least 'score' and 'entity_key'")

    original_rows = len(df)
    keep = pd.notna(df["score"])

    if "status" in df.columns:
        keep &= _nonempty_text(df["status"]).str.upper().eq("NORMAL")
    if "decision_source" in df.columns:
        keep &= _nonempty_text(df["decision_source"]).eq("model")
    if "support_sources" in df.columns:
        keep &= _nonempty_text(df["support_sources"]).eq("")
    if "phase" in df.columns:
        keep &= _nonempty_text(df["phase"]).isin(["scored", "normalizing", ""]) 
    if "reason" in df.columns:
        keep &= ~_nonempty_text(df["reason"]).str.contains("threshold hit", case=False, na=False)

    alerted_keys: set[str] = set()
    if alerts_csv is not None and alerts_csv.exists():
        alerts = pd.read_csv(alerts_csv, low_memory=False)
        for col in ("entity_key", "emit_key"):
            if col in alerts.columns:
                alerted_keys.update(v for v in _nonempty_text(alerts[col]).tolist() if v)
    if alerted_keys:
        keep &= ~_nonempty_text(df["entity_key"]).isin(alerted_keys)

    out = df.loc[keep].copy()
    out["label"] = "Normal"
    summary = {
        "scores_csv": str(scores_csv),
        "alerts_csv": str(alerts_csv) if alerts_csv is not None else "",
        "n_rows_input": int(original_rows),
        "n_alert_keys_excluded": int(len(alerted_keys)),
        "n_rows_output": int(len(out)),
        "score_mean": float(out["score"].astype(float).mean()) if len(out) else float("nan"),
        "score_p95": float(out["score"].astype(float).quantile(0.95)) if len(out) else float("nan"),
        "score_p99": float(out["score"].astype(float).quantile(0.99)) if len(out) else float("nan"),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative benign-only subset from an unlabeled runtime run.")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--alerts-csv", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    scores_csv = Path(args.scores_csv)
    alerts_csv = Path(args.alerts_csv) if args.alerts_csv else None
    out_df, summary = prepare_clean_subset(scores_csv, alerts_csv)
    if out_df.empty:
        raise ValueError(
            "No clean rows remained after filtering. Use a cleaner benign-only run before calibration."
        )
    output_csv = Path(args.output_csv) if args.output_csv else scores_csv.with_name("scores_clean_benign.csv")
    out_df.to_csv(output_csv, index=False)
    summary["output_csv"] = str(output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
