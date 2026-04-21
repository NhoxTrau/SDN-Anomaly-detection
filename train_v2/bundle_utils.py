from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .common import DEFAULT_BUNDLE_DIR, resolve_project_path


def default_rule_policy() -> dict[str, Any]:
    return {
        "min_alert_hits": 2,
        "alert_cooldown_seconds": 5.0,
        "alert_hold_seconds": 10.0,
        "score_margin": 0.5,
        "scan_window_s": 8.0,
        "scan_unique_ports": 10,
        "scan_min_hits": 2,
        "hard_packet_rate": 4000.0,
        "hard_byte_rate": 4_000_000.0,
        "hard_packet_delta": 2500.0,
        "hard_byte_delta": 2_000_000.0,
        "volumetric_min_hits": 2,
        "baseline_window_s": 60.0,
        "baseline_min_samples": 3,
        "rate_multiplier_warn": 2.5,
        "rate_multiplier_attack": 4.0,
        "allowlisted_services": {},
    }


def export_quantile_uniform_standard_stats(preprocessor: Any, output_path: str | Path) -> Path:
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    named_steps = getattr(preprocessor, "named_steps", {})
    quantile = named_steps.get("quantile")
    standard = named_steps.get("standard")
    if quantile is None or standard is None:
        raise ValueError("Expected a Pipeline with 'quantile' and 'standard' steps")
    if str(getattr(quantile, "output_distribution", "")).strip().lower() != "uniform":
        raise ValueError("Only QuantileTransformer(output_distribution='uniform') is supported")

    payload = {
        "kind": "quantile_uniform_standard_stats",
        "version": 1,
        "quantile": {
            "references": getattr(quantile, "references_").tolist(),
            "quantiles": getattr(quantile, "quantiles_").tolist(),
            "n_quantiles": int(getattr(quantile, "n_quantiles_", len(getattr(quantile, "references_", [])))),
        },
        "standard": {
            "mean": getattr(standard, "mean_").tolist(),
            "scale": getattr(standard, "scale_").tolist(),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_runtime_bundle(
    *,
    bundle_name: str,
    onnx_path: str | Path,
    metrics_path: str | Path,
    task_type: str,
    model_name: str,
    seq_len: int,
    feature_scheme: str,
    feature_names: list[str],
    preprocessing: dict[str, Any],
    thresholds: dict[str, Any],
    output_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_root = resolve_project_path(output_root or DEFAULT_BUNDLE_DIR)
    bundle_dir = output_root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = resolve_project_path(onnx_path)
    metrics_path = resolve_project_path(metrics_path)
    onnx_dst = bundle_dir / onnx_path.name
    metrics_dst = bundle_dir / metrics_path.name
    if onnx_path.resolve() != onnx_dst.resolve():
        shutil.copy2(onnx_path, onnx_dst)
    if metrics_path.resolve() != metrics_dst.resolve():
        shutil.copy2(metrics_path, metrics_dst)

    prep_kind = preprocessing.get("kind")
    prep_out: dict[str, Any] = {"kind": prep_kind}
    if prep_kind == "signed_log_robust_stats":
        scaler_stats = resolve_project_path(preprocessing["scaler_stats_path"])
        scaler_dst = bundle_dir / scaler_stats.name
        if scaler_stats.resolve() != scaler_dst.resolve():
            shutil.copy2(scaler_stats, scaler_dst)
        prep_out["scaler_stats_filename"] = scaler_dst.name
    elif prep_kind == "sklearn_pipeline":
        pkl_path = resolve_project_path(preprocessing["preprocessor_path"])
        pkl_dst = bundle_dir / pkl_path.name
        if pkl_path.resolve() != pkl_dst.resolve():
            shutil.copy2(pkl_path, pkl_dst)
        prep_out["preprocessor_filename"] = pkl_dst.name
    elif prep_kind == "quantile_uniform_standard_stats":
        stats_path = resolve_project_path(preprocessing["preprocessor_stats_path"])
        stats_dst = bundle_dir / stats_path.name
        if stats_path.resolve() != stats_dst.resolve():
            shutil.copy2(stats_path, stats_dst)
        prep_out["preprocessor_stats_filename"] = stats_dst.name
    else:
        raise ValueError(f"Unknown preprocessing kind: {prep_kind}")

    manifest = {
        "model_name": model_name,
        "task_type": task_type,
        "seq_len": int(seq_len),
        "feature_scheme": feature_scheme,
        "feature_names": feature_names,
        "onnx_filename": onnx_dst.name,
        "metrics_filename": metrics_dst.name,
        "preprocessing": prep_out,
        "thresholds": thresholds,
        "runtime_threshold_key": thresholds.get("runtime_threshold_key", "threshold"),
        "score_direction": thresholds.get("score_direction", "higher_is_attack"),
        "rule_policy": default_rule_policy(),
        "sequence_idle_timeout_s": 120.0,
    }
    if extra:
        manifest.update(extra)
    manifest_path = bundle_dir / "runtime_bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest_path
