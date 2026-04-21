from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - handled by caller
    ort = None

from ..train_v2.common import apply_transform_array, load_scaler_stats


def _startup_trace(message: str) -> None:
    if str(os.environ.get("SDN_NIDS_STARTUP_DEBUG", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    sys.stderr.write(f"[sdn-nids-startup] {message}\n")
    sys.stderr.flush()


def _load_pickle_with_numpy_compat(path: Path) -> Any:
    """Load sklearn/numpy pickles across numpy.core vs numpy._core layouts."""

    try:
        _startup_trace(f"loading preprocessor pickle: {path}")
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        _startup_trace(f"loaded preprocessor pickle: {path}")
        return obj
    except ModuleNotFoundError as exc:
        # NumPy 2.x-generated pickles may reference numpy._core, while older
        # runtime environments only expose numpy.core. Alias the module and
        # retry so existing bundles remain loadable on Mininet/Python 3.8.
        if exc.name != "numpy._core":
            raise
        _startup_trace("detected numpy._core pickle reference; retrying with numpy.core alias")
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        _startup_trace(f"loaded preprocessor pickle after numpy alias: {path}")
        return obj


class _QuantileUniformStandardPreprocessor:
    """Runtime-only replacement for sklearn QuantileTransformer + StandardScaler."""

    def __init__(self, payload: dict[str, Any]) -> None:
        quantile = dict(payload.get("quantile", {}) or {})
        standard = dict(payload.get("standard", {}) or {})
        self.references = np.asarray(quantile.get("references", []), dtype=np.float64)
        self.quantiles = np.asarray(quantile.get("quantiles", []), dtype=np.float64)
        self.mean = np.asarray(standard.get("mean", []), dtype=np.float64)
        self.scale = np.asarray(standard.get("scale", []), dtype=np.float64)
        if self.quantiles.ndim != 2:
            raise ValueError("preprocessor quantiles must be a 2D array")
        if self.references.ndim != 1:
            raise ValueError("preprocessor references must be a 1D array")
        if self.quantiles.shape[0] != self.references.shape[0]:
            raise ValueError("preprocessor references/quantiles length mismatch")
        if self.quantiles.shape[1] != self.mean.shape[0] or self.quantiles.shape[1] != self.scale.shape[0]:
            raise ValueError("preprocessor feature count mismatch")
        self.scale = np.where(np.abs(self.scale) < 1e-12, 1.0, self.scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        orig_shape = X.shape
        X_flat = np.asarray(X, dtype=np.float64).reshape(-1, self.quantiles.shape[1])
        transformed = np.empty_like(X_flat, dtype=np.float64)
        for idx in range(self.quantiles.shape[1]):
            transformed[:, idx] = np.interp(
                X_flat[:, idx],
                self.quantiles[:, idx],
                self.references,
                left=0.0,
                right=1.0,
            )
        transformed = (transformed - self.mean) / self.scale
        return transformed.reshape(orig_shape).astype(np.float32)


class InferenceEngine:
    """Thin wrapper around ONNXRuntime + preprocessing for runtime inference.

    This keeps model/session concerns out of TelemetryRuntime so the runtime can
    focus on batching, sequencing, alerting, and state publication.
    """

    def __init__(
        self,
        *,
        bundle_dir: str | Path,
        manifest: dict[str, Any],
        feature_names: list[str],
        task_type: str,
        execution_provider: str = "cpu",
    ) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for runtime inference")

        self.bundle_dir = Path(bundle_dir)
        self.manifest = dict(manifest)
        self.feature_names = list(feature_names)
        self.task_type = str(task_type)

        providers = ["CPUExecutionProvider"]
        if execution_provider.lower() == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        onnx_path = self.bundle_dir / self.manifest["onnx_filename"]
        _startup_trace(f"creating onnxruntime session: {onnx_path}")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        _startup_trace(f"created onnxruntime session: {onnx_path}")
        self.input_name = self.session.get_inputs()[0].name

        self.preprocessor_kind = str(self.manifest["preprocessing"]["kind"])
        self.scaler_center: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self.preprocessor = None
        if self.preprocessor_kind == "signed_log_robust_stats":
            scaler_stats = load_scaler_stats(self.bundle_dir / self.manifest["preprocessing"]["scaler_stats_filename"])
            self.scaler_center = np.asarray(scaler_stats["center"], dtype=np.float32)
            self.scaler_scale = np.asarray(scaler_stats["scale"], dtype=np.float32)
            self.scaler_scale = np.where(np.abs(self.scaler_scale) < 1e-12, 1.0, self.scaler_scale)
            _startup_trace("loaded signed_log_robust_stats scaler")
        elif self.preprocessor_kind == "quantile_uniform_standard_stats":
            stats_path = self.bundle_dir / self.manifest["preprocessing"]["preprocessor_stats_filename"]
            _startup_trace(f"loading preprocessor stats: {stats_path}")
            self.preprocessor = _QuantileUniformStandardPreprocessor(
                json.loads(stats_path.read_text(encoding="utf-8"))
            )
            _startup_trace(f"loaded preprocessor stats: {stats_path}")
        elif self.preprocessor_kind == "sklearn_pipeline":
            self.preprocessor = _load_pickle_with_numpy_compat(
                self.bundle_dir / self.manifest["preprocessing"]["preprocessor_filename"]
            )
        else:
            raise ValueError(f"Unsupported preprocessing kind: {self.preprocessor_kind}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        if self.preprocessor_kind == "signed_log_robust_stats":
            assert self.scaler_center is not None and self.scaler_scale is not None
            return ((apply_transform_array(X, feature_names=self.feature_names) - self.scaler_center) / self.scaler_scale).astype(np.float32)
        assert self.preprocessor is not None
        return self.preprocessor.transform(X).astype(np.float32)

    def extract_scores(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: batch})
        if self.task_type == "classifier":
            if len(outputs) >= 2:
                return np.asarray(outputs[1], dtype=np.float32).reshape(-1)
            logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        recon = np.asarray(outputs[0], dtype=np.float32)
        if recon.shape != batch.shape:
            recon = recon.reshape(batch.shape)
        errors = np.mean((recon - batch) ** 2, axis=(1, 2))
        return np.asarray(errors, dtype=np.float32).reshape(-1)
