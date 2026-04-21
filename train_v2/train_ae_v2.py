from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from .bundle_utils import write_runtime_bundle
from .common import DEFAULT_FEATURE_SCHEME, feature_names_for_scheme, resolve_project_path
from .models_ae_v2 import (
    AnomalyScoreNormalizer,
    DenoisingMLPAutoencoder,
    FeatureWeightedMSE,
    export_onnx,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x


class _AEDataset(Dataset):
    def __init__(self, X: np.ndarray) -> None:
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


def _feature_weights_from_names(feature_names: Iterable[str]) -> list[float]:
    base = {
        "packet_count": 2.0,
        "byte_count": 2.0,
        "flow_duration_s": 1.0,
        "packet_rate": 3.5,
        "byte_rate": 3.5,
        "avg_packet_size": 2.0,
        "packet_delta": 2.5,
        "byte_delta": 2.5,
        "packet_rate_delta": 3.0,
        "byte_rate_delta": 3.0,
        "src_port_norm": 0.7,
        "dst_port_norm": 0.7,
        "protocol_tcp": 0.6,
        "protocol_udp": 0.6,
        "protocol_icmp": 0.6,
        "flow_age_s": 1.0,
        "inter_poll_gap_s": 1.5,
    }
    weights = [float(base.get(name, 1.0)) for name in feature_names]
    mean_w = max(float(np.mean(weights)), 1e-6)
    return [w / mean_w for w in weights]


@torch.no_grad()
def _compute_scores(model: DenoisingMLPAutoencoder, X: np.ndarray, device: str, batch_size: int, loss_fn: FeatureWeightedMSE) -> np.ndarray:
    loader = DataLoader(_AEDataset(X), batch_size=batch_size, shuffle=False, pin_memory=(device != "cpu"))
    scores = []
    model.eval()
    for batch in loader:
        b = batch.to(device, non_blocking=True)
        recon, _ = model(b)
        scores.append(loss_fn.per_sample_score(recon, b).detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def _compute_val_loss(model: DenoisingMLPAutoencoder, loader: DataLoader, device: str, loss_fn: FeatureWeightedMSE) -> float:
    total, n = 0.0, 0
    model.eval()
    for batch in loader:
        b = batch.to(device, non_blocking=True)
        recon, _ = model(b)
        loss = loss_fn(recon, b)
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    return total / n if n else float("nan")


def _find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(scores) == 0:
        return 0.0
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s)[::-1]
    sorted_scores = s[order]
    sorted_y = y[order]
    tp_cum = np.cumsum(sorted_y)
    fp_cum = np.cumsum(1 - sorted_y)
    pos_total = int(tp_cum[-1]) if len(tp_cum) else 0
    distinct_idx = np.where(np.diff(sorted_scores))[0]
    distinct_idx = np.r_[distinct_idx, len(sorted_scores) - 1]
    tp = tp_cum[distinct_idx].astype(np.float64)
    fp = fp_cum[distinct_idx].astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(pos_total), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(sorted_scores[distinct_idx[best]])


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, fixed_fpr: float = 0.05) -> dict:
    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result = {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "recall_at_fixed_fpr": float("nan"),
        "fixed_fpr_used": fixed_fpr,
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["pr_auc"] = float(average_precision_score(y_true, scores))
        fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
        allowed = tpr_arr[fpr_arr <= fixed_fpr]
        result["recall_at_fixed_fpr"] = float(allowed.max()) if len(allowed) else 0.0
    return result


def _compute_attack_breakdown(meta_df, scores: np.ndarray, threshold: float) -> dict:
    if len(meta_df) == 0 or "label" not in meta_df.columns:
        return {}
    y_pred = (scores >= threshold).astype(int)
    breakdown = {}
    for label in sorted(set(meta_df["label"].astype(str).tolist())):
        if label == "Normal":
            continue
        mask = meta_df["label"].astype(str).to_numpy() == label
        if not np.any(mask):
            continue
        breakdown[label] = {
            "count": int(np.sum(mask)),
            "mean_score": float(np.mean(scores[mask])),
            "recall": float(np.mean(y_pred[mask] == 1)),
        }
    return breakdown


def train_ae_v2(
    dataset: dict,
    output_dir: str | Path,
    bundle_dir: str | Path,
    seq_len: int = 4,
    device: str = "cpu",
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 8e-4,
    weight_decay: float = 1e-5,
    patience: int = 12,
    target_fpr: float = 0.05,
    latent_dim: int = 16,
    dropout: float = 0.12,
    noise_std: float = 0.08,
    model_name: str = "mlp_autoencoder_insdn",
    feature_scheme: str = DEFAULT_FEATURE_SCHEME,
) -> dict:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_cols = feature_names_for_scheme(feature_scheme)
    n_features = len(feature_cols)
    if int(dataset["X_train"].shape[-1]) != n_features:
        raise ValueError("Prepared AE dataset feature count does not match feature scheme")

    feature_weights = _feature_weights_from_names(feature_cols)
    model = DenoisingMLPAutoencoder(
        n_features=n_features,
        seq_len=seq_len,
        latent_dim=latent_dim,
        noise_std=noise_std,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = FeatureWeightedMSE(feature_weights=feature_weights, reduction="mean").to(device)
    score_loss = FeatureWeightedMSE(feature_weights=feature_weights, reduction="none").to(device)

    X_train = np.asarray(dataset["X_train"], dtype=np.float32)
    X_val = np.asarray(dataset["X_val"], dtype=np.float32)
    y_val = np.asarray(dataset["y_binary_val"], dtype=np.int64)
    normal_val_mask = y_val == 0
    if not np.any(normal_val_mask):
        raise ValueError("AE validation split must contain normal samples for calibration")

    train_loader = DataLoader(_AEDataset(X_train), batch_size=batch_size, shuffle=True, pin_memory=(device != "cpu"))
    val_normal_loader = DataLoader(_AEDataset(X_val[normal_val_mask]), batch_size=batch_size, shuffle=False, pin_memory=(device != "cpu"))

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"[{model_name}] epoch {epoch}/{epochs}", leave=False):
            b = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(b)
            loss = loss_fn(recon, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * b.shape[0]
            total_n += b.shape[0]
        train_loss = total_loss / max(total_n, 1)
        val_normal_loss = _compute_val_loss(model, val_normal_loader, device, loss_fn)
        scheduler.step(val_normal_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_normal_loss": val_normal_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        print(f"[{model_name}] epoch {epoch:03d}/{epochs} train_loss={train_loss:.6f} val_normal_loss={val_normal_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}")
        if val_normal_loss < best_val:
            best_val = val_normal_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stop at epoch {epoch}; best epoch={best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    train_scores = _compute_scores(model, X_train, device, batch_size, score_loss)
    val_scores = _compute_scores(model, X_val, device, batch_size, score_loss)
    test_scores = _compute_scores(model, np.asarray(dataset["X_test"], dtype=np.float32), device, batch_size, score_loss)

    normalizer = AnomalyScoreNormalizer()
    normalizer.fit(val_scores[normal_val_mask])
    runtime_threshold = float(normalizer.threshold_at_fpr(target_fpr))
    best_f1_threshold = _find_best_f1_threshold(y_val, val_scores) if len(np.unique(y_val)) > 1 else runtime_threshold

    val_metrics = _compute_metrics(y_val, val_scores, runtime_threshold, fixed_fpr=target_fpr)
    val_metrics["best_f1_threshold"] = float(best_f1_threshold)
    test_metrics = _compute_metrics(np.asarray(dataset["y_binary_test"], dtype=np.int64), test_scores, runtime_threshold, fixed_fpr=target_fpr)
    val_metrics["attack_breakdown"] = _compute_attack_breakdown(dataset["meta_val"], val_scores, runtime_threshold)
    test_metrics["attack_breakdown"] = _compute_attack_breakdown(dataset["meta_test"], test_scores, runtime_threshold)

    checkpoint_path = output_path / f"{model_name}.pt"
    metrics_json_path = output_path / f"{model_name}.metrics.json"
    onnx_path = output_path / f"{model_name}.onnx"
    torch.save({
        "model_name": model_name,
        "task_type": "autoencoder",
        "state_dict": model.state_dict(),
        "threshold": runtime_threshold,
        "score_direction": "higher_is_attack",
        "best_epoch": best_epoch,
        "seq_len": seq_len,
        "feature_scheme": feature_scheme,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": dataset.get("config", {}),
        "feature_weights": feature_weights,
        "normalizer": normalizer.to_dict(),
        "model_info": model.get_info(),
    }, checkpoint_path)
    export_onnx(model, onnx_path, device=device)
    metrics_json_path.write_text(json.dumps({
        "model_name": model_name,
        "task_type": "autoencoder",
        "threshold": runtime_threshold,
        "best_f1_threshold": best_f1_threshold,
        "runtime_threshold_key": "threshold",
        "score_direction": "higher_is_attack",
        "seq_len": seq_len,
        "feature_scheme": feature_scheme,
        "feature_weights": feature_weights,
        "normalizer": normalizer.to_dict(),
        "model_info": model.get_info(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }, indent=2, default=str), encoding="utf-8")

    preprocessor_path = resolve_project_path(dataset.get("config", {}).get("prepared_dir", output_path)) / "scaler_v2.pkl"
    bundle_manifest = write_runtime_bundle(
        bundle_name=model_name,
        onnx_path=onnx_path,
        metrics_path=metrics_json_path,
        task_type="autoencoder",
        model_name=model_name,
        seq_len=seq_len,
        feature_scheme=feature_scheme,
        feature_names=feature_cols,
        preprocessing={"kind": "sklearn_pipeline", "preprocessor_path": preprocessor_path},
        thresholds={
            "threshold": runtime_threshold,
            "best_f1_threshold": best_f1_threshold,
            "runtime_threshold_key": "threshold",
            "score_direction": "higher_is_attack",
        },
        output_root=bundle_dir,
        extra={
            "checkpoint_filename": checkpoint_path.name,
            "ae_model_type": "denoising_mlp_autoencoder",
            "normalizer": normalizer.to_dict(),
            "model_info": model.get_info(),
        },
    )
    return {
        "model_name": model_name,
        "task_type": "autoencoder",
        "checkpoint_path": str(checkpoint_path),
        "metrics_json_path": str(metrics_json_path),
        "bundle_manifest": str(bundle_manifest),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
