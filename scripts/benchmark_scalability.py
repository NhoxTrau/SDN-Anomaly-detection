from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_alerts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_meta(items: list[str]) -> dict[str, str]:
    meta: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _derive_attack_to_alert_s(alerts: list[dict[str, Any]]) -> float | None:
    if not alerts:
        return None
    timestamps = []
    for row in alerts:
        ts = row.get("poll_timestamp") or row.get("timestamp")
        if not ts:
            continue
        try:
            if isinstance(ts, str) and ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            timestamps.append(datetime.fromisoformat(str(ts)).timestamp())
        except Exception:
            continue
    if len(timestamps) < 2:
        return None
    start_ts = min(timestamps)
    first_alert_ts = min(timestamps)
    return max(0.0, first_alert_ts - start_ts)


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    state = _read_json(run_dir / "dashboard_state.json")
    alerts = _read_alerts(run_dir / "alerts.csv")
    scores = _read_scores(run_dir / "scores.csv")
    controller_metrics = state.get("controller_metrics", {})
    avg_latency = state.get("avg_latency_breakdown", {})
    polling_stats = controller_metrics.get("polling_stats", {})
    polling_summary = controller_metrics.get("polling_summary", state.get("polling_summary", {}))

    attack_rows = [row for row in alerts if str(row.get("status", "")).upper() == "ATTACK"]
    suspect_rows = [row for row in alerts if str(row.get("status", "")).upper() == "SUSPECT"]
    score_values = [_safe_float(row.get("score")) for row in scores if row.get("score") not in (None, "")]

    return {
        "run_id": run_dir.name,
        "meta": meta,
        "status": state.get("status", ""),
        "phase": state.get("phase", ""),
        "model_name": state.get("model_name", ""),
        "poll_interval_s": _safe_float(state.get("poll_interval_s"), _safe_float(meta.get("poll_interval_s"), 0.0)),
        "feature_state_count": _safe_int(controller_metrics.get("feature_state_count", 0)),
        "raw_stats_queue_depth": _safe_int(controller_metrics.get("raw_stats_queue_depth", 0)),
        "queue_depth": _safe_int(state.get("queue_depth", 0)),
        "alerts_total": len(alerts),
        "alerts_attack": len(attack_rows),
        "alerts_suspect": len(suspect_rows),
        "avg_total_e2e_ms": _safe_float(avg_latency.get("total_e2e_ms", 0.0)),
        "avg_switch_latency_ms": _safe_float(avg_latency.get("switch_latency_ms", 0.0)),
        "avg_inference_ms": _safe_float(avg_latency.get("inference_ms", 0.0)),
        "avg_reply_delay_ms": _safe_float(polling_summary.get("avg_reply_delay_ms", 0.0)),
        "max_reply_delay_ms": _safe_float(polling_summary.get("max_reply_delay_ms", 0.0)),
        "timeout_total": _safe_int(polling_summary.get("timeout_total", 0)),
        "trimmed_flows_total": _safe_int(polling_summary.get("trimmed_flows_total", 0)),
        "throughput_obs_s": _safe_float(state.get("throughput_obs_s", 0.0)),
        "dropped_rows": _safe_int(state.get("dropped_rows", 0)),
        "csv_drop_count": _safe_int(state.get("csv_drop_count", 0)),
        "stream_drop_count": _safe_int(state.get("stream_drop_count", 0)),
        "recommended_poll_interval_s": _safe_float(state.get("recommended_poll_interval_s", controller_metrics.get("recommended_poll_interval_s", 0.0))),
        "attack_to_alert_s": _derive_attack_to_alert_s(attack_rows),
        "peak_score": max(score_values) if score_values else 0.0,
    }


def _score_row(row: dict[str, Any]) -> float:
    """Lower is better. Balanced for thesis demo stability over raw aggressiveness."""
    latency = _safe_float(row.get("avg_total_e2e_ms"))
    timeout_penalty = _safe_int(row.get("timeout_total")) * 150.0
    drop_penalty = (_safe_int(row.get("dropped_rows")) + _safe_int(row.get("csv_drop_count")) + _safe_int(row.get("stream_drop_count"))) * 2.0
    trim_penalty = _safe_int(row.get("trimmed_flows_total")) * 0.05
    throughput_bonus = _safe_float(row.get("throughput_obs_s")) * 0.30
    return latency + timeout_penalty + drop_penalty + trim_penalty - throughput_bonus


def _group_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get("meta", {}).get(key, row.get(key))
        groups[str(value)].append(row)
    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        out.append({
            "group": group_key,
            "runs": [r["run_id"] for r in group_rows],
            "count": len(group_rows),
            "avg_total_e2e_ms": mean(_safe_float(r.get("avg_total_e2e_ms")) for r in group_rows),
            "avg_reply_delay_ms": mean(_safe_float(r.get("avg_reply_delay_ms")) for r in group_rows),
            "avg_throughput_obs_s": mean(_safe_float(r.get("throughput_obs_s")) for r in group_rows),
            "timeout_total": sum(_safe_int(r.get("timeout_total")) for r in group_rows),
            "trimmed_flows_total": sum(_safe_int(r.get("trimmed_flows_total")) for r in group_rows),
            "avg_recommended_poll_interval_s": mean(_safe_float(r.get("recommended_poll_interval_s")) for r in group_rows),
        })
    return out


def build_report(runtime_root: Path, runs: list[str], metas: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = []
    for run in runs:
        run_dir = runtime_root / run
        if not run_dir.exists():
            continue
        rows.append(summarize_run(run_dir, metas.get(run, {})))

    ranked = sorted(rows, key=_score_row)
    best = ranked[0] if ranked else {}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_root": str(runtime_root),
        "runs": rows,
        "best_run": best,
        "recommended_poll_interval_s": best.get("recommended_poll_interval_s") or best.get("poll_interval_s"),
        "analysis": {
            "by_poll_interval": _group_by(rows, "poll_interval_s"),
            "by_hosts": _group_by(rows, "hosts"),
            "by_scenario": _group_by(rows, "scenario"),
        },
    }


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        meta = dict(row.get("meta", {}))
        flat = {k: v for k, v in row.items() if k != "meta"}
        for mk, mv in meta.items():
            flat[f"meta_{mk}"] = mv
        flat_rows.append(flat)
    fieldnames = sorted({key for row in flat_rows for key in row.keys()})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_markdown(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    best = report.get("best_run", {}) or {}
    analysis = report.get("analysis", {}) or {}
    lines = [
        "# Scalability Benchmark Summary",
        "",
        f"Generated at: `{report.get('generated_at', '')}`",
        "",
    ]
    if best:
        lines.extend([
            "## Recommended operating point",
            "",
            f"- Best run: `{best.get('run_id', '')}`",
            f"- Poll interval: `{best.get('poll_interval_s', '—')}` s",
            f"- Recommended interval: `{best.get('recommended_poll_interval_s', '—')}` s",
            f"- Avg end-to-end latency: `{best.get('avg_total_e2e_ms', '—')}` ms",
            f"- Avg reply delay: `{best.get('avg_reply_delay_ms', '—')}` ms",
            f"- Throughput: `{best.get('throughput_obs_s', '—')}` obs/s",
            f"- Timeouts: `{best.get('timeout_total', '—')}`",
            "",
        ])
    for title, rows in [("By poll interval", analysis.get("by_poll_interval", [])), ("By host count", analysis.get("by_hosts", [])), ("By scenario", analysis.get("by_scenario", []))]:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["No grouped data.", ""])
            continue
        lines.extend([
            "| Group | Runs | Avg E2E ms | Avg reply ms | Avg throughput | Timeouts | Trimmed flows | Avg recommended poll |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row.get('group', '—')} | {row.get('count', 0)} | {row.get('avg_total_e2e_ms', 0.0):.2f} | {row.get('avg_reply_delay_ms', 0.0):.2f} | {row.get('avg_throughput_obs_s', 0.0):.2f} | {row.get('timeout_total', 0)} | {row.get('trimmed_flows_total', 0)} | {row.get('avg_recommended_poll_interval_s', 0.0):.2f} |"
            )
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate runtime runs into a scalability report artifact.")
    parser.add_argument("--runtime-root", default="runtime_logs")
    parser.add_argument("--run", action="append", default=[], help="Run ID to include. Repeatable. Default: all runs in runtime root.")
    parser.add_argument("--meta", action="append", default=[], help="Attach metadata to a run: run_id:key=value,key=value")
    parser.add_argument("--output", default="artifacts_v4/benchmark/scalability_report.json")
    parser.add_argument("--csv-output", default="artifacts_v4/benchmark/scalability_report.csv")
    parser.add_argument("--markdown-output", default="artifacts_v4/benchmark/scalability_summary.md")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    runs = args.run or (sorted([item.name for item in runtime_root.iterdir() if item.is_dir()], reverse=True) if runtime_root.exists() else [])
    metas: dict[str, dict[str, str]] = {}
    for item in args.meta:
        if ":" not in item:
            continue
        run_id, raw_meta = item.split(":", 1)
        metas[run_id.strip()] = _parse_meta([part for part in raw_meta.split(",") if part.strip()])

    report = build_report(runtime_root, runs, metas)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["output_path"] = str(output.resolve())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report.get("runs", []), Path(args.csv_output))
    write_markdown(report, Path(args.markdown_output))
    print(f"Wrote scalability report to {output}")
    print(f"Wrote flat CSV to {args.csv_output}")
    print(f"Wrote markdown summary to {args.markdown_output}")


if __name__ == "__main__":
    main()
