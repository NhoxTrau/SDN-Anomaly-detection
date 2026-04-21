from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torch.amp import GradScaler as TorchGradScaler, autocast as torch_autocast

    def _make_grad_scaler(enabled: bool):
        try:
            return TorchGradScaler("cuda", enabled=enabled)
        except TypeError:
            return TorchGradScaler(enabled=enabled)

    def _autocast_cuda(enabled: bool = True):
        try:
            return torch_autocast("cuda", enabled=enabled)
        except TypeError:
            return torch_autocast(enabled=enabled)
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast

    def _make_grad_scaler(enabled: bool):
        return CudaGradScaler(enabled=enabled)

    def _autocast_cuda(enabled: bool = True):
        return cuda_autocast(enabled=enabled)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from .bundle_utils import write_runtime_bundle
from .common import (
    ATTACK_LABELS,
    DEFAULT_BUNDLE_DIR,
    DEFAULT_INSDN_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_PREPARED_DIR,
    RANDOM_SEED,
    feature_names_for_scheme,
    resolve_project_path,
)
from .models import export_onnx, get_model, get_model_task
from .prepare_data import DEFAULT_SEQ_LEN, load_prepared, prepare_data


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        if self.y is None:
            return self.X[index]
        return self.X[index], self.y[index]


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def _build_balanced_sampler(y_binary: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(y_binary.astype(np.int64), minlength=2).astype(np.float64)
    class_counts = np.where(class_counts <= 0.0, 1.0, class_counts)
    weights = 1.0 / class_counts[y_binary.astype(np.int64)]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def _configure_balance(y_binary: np.ndarray, balance_mode: str) -> tuple[str, Optional[torch.Tensor]]:
    pos_count = max(int(np.sum(y_binary == 1)), 1)
    neg_count = max(int(np.sum(y_binary == 0)), 1)
    pos_ratio = pos_count / max(pos_count + neg_count, 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    if balance_mode == "auto":
        if pos_ratio < 0.10:
            return "both", pos_weight
        if pos_ratio < 0.25:
            return "pos_weight", pos_weight
        return "none", None
    if balance_mode == "none":
        return "none", None
    if balance_mode == "sampler":
        return "sampler", None
    if balance_mode == "pos_weight":
        return "pos_weight", pos_weight
    if balance_mode == "both":
        return "both", pos_weight
    raise ValueError(f"Unknown balance mode: {balance_mode}")


def _predict_from_scores(scores: np.ndarray, threshold: float, score_direction: str) -> np.ndarray:
    if score_direction == "lower_is_attack":
        return (scores <= threshold).astype(np.int64)
    return (scores >= threshold).astype(np.int64)


def _find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray, score_direction: str) -> float:
    if len(scores) == 0:
        return 0.0
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if score_direction == "lower_is_attack":
        s = -s
    order = np.argsort(s)[::-1]
    sorted_scores = s[order]
    sorted_y = y[order]
    tp_cum = np.cumsum(sorted_y)
    fp_cum = np.cumsum(1 - sorted_y)
    pos_total = int(tp_cum[-1]) if len(tp_cum) else 0
    neg_total = int(fp_cum[-1]) if len(fp_cum) else 0
    distinct_idx = np.where(np.diff(sorted_scores))[0]
    distinct_idx = np.r_[distinct_idx, len(sorted_scores) - 1]
    tp = tp_cum[distinct_idx].astype(np.float64)
    fp = fp_cum[distinct_idx].astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(pos_total), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    fpr = fp / max(float(neg_total), 1.0)
    best = int(np.argmax(np.stack([f1, -fpr], axis=1), axis=0)[0])
    threshold = float(sorted_scores[distinct_idx[best]])
    return -threshold if score_direction == "lower_is_attack" else threshold


def select_threshold(train_scores: np.ndarray, val_scores: np.ndarray, y_val: np.ndarray, target_fpr: float, strategy: str, score_direction: str) -> float:
    if len(val_scores) > 0 and np.sum(y_val == 0) > 0:
        if strategy == "best_f1" and len(np.unique(y_val)) > 1:
            return _find_best_f1_threshold(y_val, val_scores, score_direction)
        normal_scores = val_scores[y_val == 0]
        q = min(max(1.0 - target_fpr, 0.0), 1.0)
        if score_direction == "lower_is_attack":
            q = 1.0 - q
        return float(np.quantile(normal_scores, q))
    if len(train_scores) > 0:
        q = 0.995 if score_direction == "higher_is_attack" else 0.005
        return float(np.quantile(train_scores, q))
    return 0.0


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, fixed_fpr: float = 0.05, score_direction: str = "higher_is_attack") -> dict[str, float]:
    y_pred = _predict_from_scores(scores, threshold, score_direction)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc_scores = -scores if score_direction == "lower_is_attack" else scores
    result = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "recall_at_fixed_fpr": float("nan"),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = float(roc_auc_score(y_true, roc_scores))
        result["pr_auc"] = float(average_precision_score(y_true, roc_scores))
        fpr_arr, tpr_arr, _ = roc_curve(y_true, roc_scores)
        allowed = tpr_arr[fpr_arr <= fixed_fpr]
        result["recall_at_fixed_fpr"] = float(allowed.max()) if len(allowed) else 0.0
    return result


def compute_attack_breakdown(meta_df, scores: np.ndarray, threshold: float, score_direction: str) -> dict[str, dict[str, float]]:
    if len(meta_df) == 0 or "label" not in meta_df.columns:
        return {}
    y_pred = _predict_from_scores(scores, threshold, score_direction)
    breakdown: dict[str, dict[str, float]] = {}
    for label in ATTACK_LABELS[1:]:
        mask = meta_df["label"].astype(str).to_numpy() == label
        if not np.any(mask):
            continue
        breakdown[label] = {
            "count": int(np.sum(mask)),
            "mean_score": float(np.mean(scores[mask])),
            "recall": float(np.mean(y_pred[mask] == 1)),
        }
    return breakdown


def compute_scores(model: nn.Module, X: np.ndarray, device: str, batch_size: int, task_type: str, num_workers: int = 0) -> np.ndarray:
    loader = DataLoader(SequenceDataset(X), batch_size=batch_size, shuffle=False, pin_memory=(device != "cpu"), num_workers=max(int(num_workers), 0), persistent_workers=(num_workers > 0))
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = batch.to(device, non_blocking=True)
            res = model.predict_scores(b) if task_type == "classifier" else model.anomaly_score(b)
            scores.append(res.detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)


def compute_loss_over_loader(model: nn.Module, loader: DataLoader, device: str, task_type: str, loss_fn: nn.Module) -> float:
    total_loss, total_samples = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if task_type == "classifier":
                X_batch, y_batch = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                logits, _ = model(X_batch)
                loss = loss_fn(logits, y_batch)
                n = X_batch.shape[0]
            else:
                X_batch = batch.to(device, non_blocking=True) if not isinstance(batch, (tuple, list)) else batch[0].to(device, non_blocking=True)
                recon, _ = model(X_batch)
                loss = loss_fn(recon, X_batch)
                n = X_batch.shape[0]
            total_loss += loss.item() * n
            total_samples += n
    return total_loss / total_samples if total_samples > 0 else float("nan")


def _build_schedulers(optimizer: torch.optim.Optimizer, warmup_epochs: int = 3, plateau_patience: int = 5, plateau_factor: float = 0.3, min_lr: float = 1e-7) -> tuple:
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=plateau_factor, patience=plateau_patience, min_lr=min_lr)
    return warmup, plateau


def _reset_adam_momentum(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state:
                state = optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key].mul_(0.0)


def train_one_classifier(
    model_name: str,
    dataset: dict,
    output_dir: str | Path,
    seq_len: int,
    bundle_dir: str | Path,
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    target_fpr: float = 0.05,
    threshold_strategy: str = "target_fpr",
    label_smoothing: float = 0.05,
    warmup_epochs: int = 3,
    grad_accum_steps: int = 1,
    train_num_workers: int = 0,
    eval_num_workers: int = 0,
    balance_mode: str = "auto",
    use_ema: bool = True,
    use_amp: bool = True,
    feature_scheme: str = "telemetry_v2",
) -> dict:
    use_amp = use_amp and device != "cpu" and torch.cuda.is_available()
    scaler = _make_grad_scaler(enabled=use_amp) if use_amp else None
    n_features = int(dataset["X_train"].shape[-1])
    model = get_model(model_name, n_features=n_features, seq_len=seq_len).to(device)
    ema = ModelEMA(model, decay=0.999) if use_ema else None
    y_train = np.asarray(dataset["y_binary_train"], dtype=np.int64)
    y_val = np.asarray(dataset["y_binary_val"], dtype=np.int64)
    balance_mode, pos_weight = _configure_balance(y_train, balance_mode)
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_weight_device = pos_weight.to(device) if pos_weight is not None else None
    loss_fn = LabelSmoothingBCE(smoothing=label_smoothing, pos_weight=pos_weight_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    warmup_sched, plateau_sched = _build_schedulers(optimizer, warmup_epochs=warmup_epochs)

    train_loader_kwargs = dict(batch_size=batch_size, pin_memory=(device != "cpu"), num_workers=max(int(train_num_workers), 0), persistent_workers=(train_num_workers > 0))
    if balance_mode in {"sampler", "both"} and len(np.unique(y_train)) > 1:
        train_loader = DataLoader(SequenceDataset(dataset["X_train"], y_train), sampler=_build_balanced_sampler(y_train), **train_loader_kwargs)
    else:
        train_loader = DataLoader(SequenceDataset(dataset["X_train"], y_train), shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(SequenceDataset(dataset["X_val"], y_val), batch_size=batch_size, shuffle=False, pin_memory=(device != "cpu"), num_workers=max(int(eval_num_workers), 0), persistent_workers=(eval_num_workers > 0))

    print(f"\n{'=' * 72}\n[{model_name}] Training | features={n_features} | seq_len={seq_len}")
    print(f"[{model_name}] train windows={len(y_train):,} normal={neg_count:,} attack={pos_count:,} balance_mode={balance_mode} batch_size={batch_size}")
    print(f"{'=' * 72}")

    history = []
    best_state, best_epoch, best_val_f1, best_val_loss, epochs_without_improve = None, 0, -1.0, float("inf"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_n = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"[{model_name}] epoch {epoch:03d}/{epochs} TRAIN", leave=False)
        for step, (X_batch, y_batch) in enumerate(train_bar):
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            if use_amp:
                with _autocast_cuda(enabled=True):
                    logits, _ = model(X_batch)
                    loss = loss_fn(logits, y_batch) / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits, _ = model(X_batch)
                loss = loss_fn(logits, y_batch) / grad_accum_steps
                loss.backward()
            running_loss += loss.item() * grad_accum_steps * X_batch.shape[0]
            running_n += X_batch.shape[0]
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        train_loss = running_loss / max(running_n, 1)
        if epoch <= warmup_epochs:
            warmup_sched.step()
        if ema is not None:
            ema.apply_shadow(model)
        val_loss = compute_loss_over_loader(model, val_loader, device, "classifier", loss_fn)
        val_scores = compute_scores(model, dataset["X_val"], device, batch_size, "classifier", num_workers=eval_num_workers)
        val_f1_threshold = _find_best_f1_threshold(y_val, val_scores, "higher_is_attack")
        val_f1_pred = _predict_from_scores(val_scores, val_f1_threshold, "higher_is_attack")
        _, _, val_f1, _ = precision_recall_fscore_support(y_val, val_f1_pred, average="binary", zero_division=0)
        if ema is not None:
            ema.restore(model)
        old_lr = float(optimizer.param_groups[0]["lr"])
        if epoch > warmup_epochs:
            plateau_sched.step(val_loss if not np.isnan(val_loss) else train_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr < old_lr * 0.95:
            _reset_adam_momentum(optimizer)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": float(val_f1), "lr": new_lr})
        print(f"[{model_name}] epoch {epoch:03d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  val_f1={val_f1:.4f}  lr={new_lr:.2e}")
        if val_f1 > best_val_f1 or (np.isclose(val_f1, best_val_f1, atol=1e-4) and not np.isnan(val_loss) and val_loss < best_val_loss):
            best_val_f1, best_val_loss, best_epoch = float(val_f1), val_loss if not np.isnan(val_loss) else best_val_loss, epoch
            if ema is not None:
                ema.apply_shadow(model)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                ema.restore(model)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print(f"[{model_name}] Early stop at epoch {epoch}. Best epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    train_scores = compute_scores(model, dataset["X_train"], device, batch_size, "classifier", num_workers=eval_num_workers)
    val_scores = compute_scores(model, dataset["X_val"], device, batch_size, "classifier", num_workers=eval_num_workers)
    test_scores = compute_scores(model, dataset["X_test"], device, batch_size, "classifier", num_workers=eval_num_workers)
    score_direction = "higher_is_attack"
    threshold = select_threshold(train_scores, val_scores, y_val, target_fpr, threshold_strategy, score_direction)
    val_metrics = compute_metrics(y_val, val_scores, threshold, target_fpr, score_direction)
    test_metrics = compute_metrics(np.asarray(dataset["y_binary_test"], dtype=np.int64), test_scores, threshold, target_fpr, score_direction)
    val_metrics["attack_breakdown"] = compute_attack_breakdown(dataset["meta_val"], val_scores, threshold, score_direction)
    test_metrics["attack_breakdown"] = compute_attack_breakdown(dataset["meta_test"], test_scores, threshold, score_direction)

    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{model_name}.pt"
    metrics_json_path = output_path / f"{model_name}.metrics.json"
    onnx_path = output_path / f"{model_name}.onnx"
    torch.save({
        "model_name": model_name,
        "task_type": "classifier",
        "state_dict": model.state_dict(),
        "threshold": threshold,
        "score_direction": score_direction,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "seq_len": seq_len,
        "feature_scheme": feature_scheme,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": dataset.get("config", {}),
    }, checkpoint_path)
    export_onnx(model, str(onnx_path), device=device)
    metrics_json_path.write_text(json.dumps({
        "model_name": model_name,
        "task_type": "classifier",
        "threshold": threshold,
        "score_direction": score_direction,
        "seq_len": seq_len,
        "feature_scheme": feature_scheme,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "balance_mode": balance_mode,
    }, indent=2, default=str), encoding="utf-8")
    scaler_stats_path = resolve_project_path(dataset.get("config", {}).get("prepared_dir", DEFAULT_PREPARED_DIR)) / "scaler_stats.json"
    bundle_manifest = write_runtime_bundle(
        bundle_name=model_name,
        onnx_path=onnx_path,
        metrics_path=metrics_json_path,
        task_type="classifier",
        model_name=model_name,
        seq_len=seq_len,
        feature_scheme=feature_scheme,
        feature_names=feature_names_for_scheme(feature_scheme),
        preprocessing={"kind": "signed_log_robust_stats", "scaler_stats_path": scaler_stats_path},
        thresholds={"threshold": threshold, "runtime_threshold_key": "threshold", "score_direction": score_direction},
        output_root=bundle_dir,
        extra={"checkpoint_filename": checkpoint_path.name},
    )
    return {
        "model_name": model_name,
        "task_type": "classifier",
        "checkpoint_path": str(checkpoint_path),
        "metrics_json_path": str(metrics_json_path),
        "bundle_manifest": str(bundle_manifest),
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }


def _ensure_prepared_dataset(args) -> dict:
    prepared_path = resolve_project_path(args.prepared_dir)
    dataset_dir = resolve_project_path(args.dataset_dir)
    if not args.force_prepare and prepared_path.exists() and (prepared_path / "train.npz").exists():
        print(f"Loading prepared data from {prepared_path}")
        dataset = load_prepared(prepared_path)
    else:
        print(f"Preparing InSDN sequences from {dataset_dir} ...")
        dataset = prepare_data(
            dataset_dir=dataset_dir,
            save_dir=prepared_path,
            seq_len=args.seq_len,
            stride=args.stride,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            split_gap=args.split_gap,
            feature_scheme=args.feature_scheme,
        )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenFlow-safe sequence classifiers on the bundled InSDN dataset.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_INSDN_DIR))
    parser.add_argument("--prepared-dir", default=str(DEFAULT_PREPARED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--threshold-strategy", choices=["best_f1", "target_fpr"], default="target_fpr")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--balance-mode", choices=["auto", "none", "sampler", "pos_weight", "both"], default="auto")
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--feature-scheme", choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"], default="insdn_ml_core_v1")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--convert-first", action="store_true", help="deprecated; ignored for InSDN pipeline")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--convert-chunksize", type=int, default=300_000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-gap", type=int, default=1)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()
    set_seed()
    dataset = _ensure_prepared_dataset(args)
    seq_len = int(dataset.get("config", {}).get("seq_len", args.seq_len))
    print(f"Dataset loaded: train={len(dataset['X_train'])}, val={len(dataset['X_val'])}, test={len(dataset['X_test'])}, features={dataset['X_train'].shape[-1]}, seq_len={seq_len}")
    result = train_one_classifier(
        model_name=args.model,
        dataset=dataset,
        output_dir=args.output_dir,
        bundle_dir=args.bundle_dir,
        seq_len=seq_len,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        target_fpr=args.target_fpr,
        threshold_strategy=args.threshold_strategy,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        grad_accum_steps=args.grad_accum_steps,
        train_num_workers=args.train_num_workers,
        eval_num_workers=args.eval_num_workers,
        balance_mode=args.balance_mode,
        use_ema=not args.no_ema,
        use_amp=not args.no_amp,
        feature_scheme=str(dataset.get("config", {}).get("feature_scheme", args.feature_scheme)),
    )
    summary_path = resolve_project_path(args.output_dir) / "training_summary.json"
    summary_path.write_text(json.dumps([result], indent=2, default=str), encoding="utf-8")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
