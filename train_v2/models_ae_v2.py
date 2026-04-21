"""
models_ae_v2.py — Improved MLP Autoencoder for SDN anomaly detection.

Sử dụng telemetry_v2 (15 features) — không có feature duplicates.

Thay đổi so với phiên bản gốc:
  1. Feature weights cập nhật cho telemetry_v2 (15 features).
     Phiên bản gốc có bugs: packet_rate_delta = packet_rate (duplicate), nên
     effective weight của packet_rate là 3.5+3.5=7.0 — model bị dominated bởi
     1 feature. Với telemetry_v2, mỗi feature thực sự khác nhau nên weights
     phản ánh đúng tầm quan trọng.
  2. DenoisingMLPAutoencoder: n_features default=15, input_dim=90 (đúng).
  3. AnomalyScoreNormalizer: fixed threshold_at_fpr() — không dùng Gaussian
     approximation nữa (AE reconstruction errors không follow Gaussian).
     Thay vào đó store full percentile interpolation data.
  4. Architecture docstring được sửa lại đúng với thực tế.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Feature weight map — insdn_openflow_v1 / telemetry_v2 alias (15 features)
#
# Feature order (index 0–14):
#   0  packet_count
#   1  byte_count
#   2  flow_duration_s
#   3  packet_rate
#   4  byte_rate
#   5  avg_packet_size
#   6  packet_delta
#   7  byte_delta
#   8  packet_rate_delta
#   9  byte_rate_delta
#  10  src_port_norm
#  11  dst_port_norm
#  12  protocol_tcp
#  13  protocol_udp
#  14  protocol_icmp
# ---------------------------------------------------------------------------
TELEMETRY_FEATURE_WEIGHTS_V2 = [
    2.0, 2.0, 1.0, 3.5, 3.5, 2.0, 2.5, 2.5, 3.0, 3.0, 0.7, 0.7, 0.6, 0.6, 0.6,
]
assert len(TELEMETRY_FEATURE_WEIGHTS_V2) == 15, "Weight list must have exactly 15 entries for telemetry_v2"

# Normalize to mean=1 so total loss scale stays roughly constant vs unweighted MSE
_w_v2 = TELEMETRY_FEATURE_WEIGHTS_V2
_mean_w_v2 = sum(_w_v2) / len(_w_v2)
TELEMETRY_FEATURE_WEIGHTS_V2_NORM = [x / _mean_w_v2 for x in _w_v2]

# Legacy weights for telemetry_v1 (13 features) — kept for backward compatibility
# NOTE: these weights are misleading because v1 has feature duplication.
TELEMETRY_FEATURE_WEIGHTS_V1 = [
    1.0,  # flow_duration_s
    2.0,  # packet_count
    2.0,  # byte_count
    3.5,  # packet_rate
    3.5,  # byte_rate
    2.0,  # avg_packet_size
    3.0,  # packet_delta (== packet_count in v1 — duplicate)
    3.0,  # byte_delta (== byte_count in v1 — duplicate)
    3.5,  # packet_rate_delta (== packet_rate in v1 — duplicate)
    3.5,  # byte_rate_delta (== byte_rate in v1 — duplicate)
    1.0,  # flow_age_s (== flow_duration_s in v1 — duplicate)
    1.5,  # inter_poll_gap_s (constant in v1 — zero info)
    0.4,  # dst_port_norm
]
_w_v1 = TELEMETRY_FEATURE_WEIGHTS_V1
_mean_w_v1 = sum(_w_v1) / len(_w_v1)
TELEMETRY_FEATURE_WEIGHTS_V1_NORM = [x / _mean_w_v1 for x in _w_v1]

# Default to v2
TELEMETRY_FEATURE_WEIGHTS_NORM = TELEMETRY_FEATURE_WEIGHTS_V2_NORM


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
class FeatureWeightedMSE(nn.Module):
    """MSE loss with per-feature weights.

    Args:
        feature_weights: list or tensor of shape (n_features,).
            Default: TELEMETRY_FEATURE_WEIGHTS_V2_NORM (15-feature telemetry_v2).
        reduction: 'mean' for training loss, 'none' for per-sample score.
    """
    def __init__(
        self,
        feature_weights: Optional[list[float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        weights = feature_weights or TELEMETRY_FEATURE_WEIGHTS_V2_NORM
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # recon / target: (B, seq_len, n_features) or (B, n_features)
        sq = (recon - target) ** 2
        weighted = sq * self.weights   # broadcast along last dim
        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        else:  # 'none' → per-sample scalar
            return weighted.mean(dim=tuple(range(1, weighted.ndim)))

    def per_sample_score(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return anomaly score (scalar) per sample in the batch."""
        sq = (recon - target) ** 2
        weighted = sq * self.weights
        return weighted.mean(dim=tuple(range(1, weighted.ndim)))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# Improved MLP Autoencoder (Denoising)
# ---------------------------------------------------------------------------
class DenoisingMLPAutoencoder(nn.Module):
    """Denoising MLP Autoencoder for SDN flow anomaly detection.

    Architecture (default telemetry_v2: n_features=15, seq_len=6):
        input_dim = 15 * 6 = 90
        Encoder: 90 → 128 [ResBlock 256] → 48 [ResBlock 96] → latent_dim (Tanh)
        Decoder: latent_dim → 48 [ResBlock 96] → 128 [ResBlock 256] → 90
        Compression: 90 / 16 = 5.625x

    Key design choices:
        - Tight bottleneck (latent_dim=16, 5.6x compression): forces the AE to
          specialise on the normal-traffic manifold. Attack samples produce higher
          reconstruction error because they lie off this manifold.
        - Denoising: Gaussian noise added to input during training. The AE learns
          to denoise, which improves robustness and prevents memorisation of the
          training set.
        - MLP (not LSTM/Transformer): deliberate choice for real-time deployment.
          MLP inference latency is ~0.1–0.5 ms vs ~5–20 ms for LSTM on CPU,
          crucial for SDN controller performance.
        - Feature-weighted loss: rate and delta features (highly discriminative
          for traffic anomalies) receive higher weight.

    Args:
        n_features: number of features per timestep. 15 for telemetry_v2 (default),
            13 for legacy telemetry_v1.
        seq_len: sequence length (number of consecutive polling observations). Default 6.
        latent_dim: bottleneck dimension. Smaller = higher compression = better anomaly
            separation. Recommended: 12–20 for telemetry_v2.
        noise_std: std of Gaussian noise added during training (denoising regularization).
            Set to 0.0 to disable. Recommended: 0.05–0.15.
        dropout: dropout rate in residual blocks.
    """
    def __init__(
        self,
        n_features: int = 15,       # 15 for telemetry_v2; 13 for legacy telemetry_v1
        seq_len: int = 6,
        latent_dim: int = 16,
        noise_std: float = 0.08,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        input_dim = n_features * seq_len   # e.g. 15*6=90 for telemetry_v2

        # ---- Encoder ----
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.enc_res1 = ResidualBlock(128, 256, dropout=dropout)
        self.enc_down1 = nn.Sequential(
            nn.Linear(128, 48),
            nn.LayerNorm(48),
            nn.GELU(),
        )
        self.enc_res2 = ResidualBlock(48, 96, dropout=dropout)
        self.enc_down2 = nn.Sequential(
            nn.Linear(48, latent_dim),
            nn.Tanh(),             # bounded [-1, 1] latent space
        )

        # ---- Decoder ----
        self.dec_up1 = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
        )
        self.dec_res1 = ResidualBlock(48, 96, dropout=dropout)
        self.dec_up2 = nn.Sequential(
            nn.Linear(48, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dec_res2 = ResidualBlock(128, 256, dropout=dropout)
        self.dec_out = nn.Linear(128, input_dim)

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def encode(self, x_flat: torch.Tensor) -> torch.Tensor:
        h = self.enc_proj(x_flat)
        h = self.enc_res1(h)
        h = self.enc_down1(h)
        h = self.enc_res2(h)
        return self.enc_down2(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_up1(z)
        h = self.dec_res1(h)
        h = self.dec_up2(h)
        h = self.dec_res2(h)
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len, n_features)
        Returns:
            recon: (B, seq_len, n_features) — reconstructed clean input
            latent: (B, latent_dim)
        """
        b = x.shape[0]
        x_flat = x.reshape(b, -1)            # (B, n_features*seq_len)
        x_noisy = self._add_noise(x_flat)    # denoising: train with noisy, predict clean
        z = self.encode(x_noisy)             # (B, latent_dim)
        recon_flat = self.decode(z)          # (B, n_features*seq_len)
        recon = recon_flat.reshape(b, self.seq_len, self.n_features)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Unweighted MSE score — for inference (no noise applied)."""
        self.eval()
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))

    def weighted_anomaly_score(
        self,
        x: torch.Tensor,
        loss_fn: Optional[FeatureWeightedMSE] = None,
    ) -> torch.Tensor:
        """Feature-weighted anomaly score — better attack discrimination."""
        self.eval()
        recon, _ = self.forward(x)
        if loss_fn is not None:
            return loss_fn.per_sample_score(recon, x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))

    def get_info(self) -> dict[str, object]:
        n_params = sum(p.numel() for p in self.parameters())
        input_dim = self.n_features * self.seq_len
        return {
            "model": "DenoisingMLP-AE",
            "n_params": n_params,
            "size_mb": round(n_params * 4 / 1024 / 1024, 3),
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "input_dim": input_dim,
            "latent_dim": self.latent_dim,
            "compression_ratio": round(input_dim / self.latent_dim, 2),
            "denoising": self.noise_std > 0.0,
            "noise_std": self.noise_std,
        }


# ---------------------------------------------------------------------------
# Score normalisation helper
# ---------------------------------------------------------------------------
class AnomalyScoreNormalizer:
    """Normalise anomaly scores based on normal-traffic distribution at val set.

    Approach:
        1. Collect reconstruction scores of NORMAL samples from val set.
        2. Store mean, std, and a set of percentile thresholds.
        3. At inference: z_score = (score - normal_mean) / normal_std
           OR use raw threshold at target FPR.

    FIX vs original:
        threshold_at_fpr() previously fell back to a Gaussian approximation
        (normal_mean + k*normal_std) for FPRs other than 0.05 and 0.01.
        AE reconstruction errors are typically right-skewed (NOT Gaussian), so
        this approximation is unreliable. The fix: store and interpolate from
        actual percentiles computed at fit time.
    """
    def __init__(self) -> None:
        self.normal_mean: float = 0.0
        self.normal_std: float = 1.0
        # Stored percentile thresholds at common FPR targets
        self._percentiles: dict[float, float] = {}
        # Fine-grained CDF for arbitrary FPR interpolation
        self._cdf_scores: list[float] = []
        self._cdf_percentiles: list[float] = []

    def fit(self, normal_scores: np.ndarray) -> None:
        """Fit on reconstruction scores of normal val samples."""
        if len(normal_scores) == 0:
            return
        scores = np.asarray(normal_scores, dtype=np.float64)
        self.normal_mean = float(np.mean(scores))
        self.normal_std = float(np.std(scores)) + 1e-9
        # Pre-compute percentiles at common FPR targets
        for fpr in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
            pct = (1.0 - fpr) * 100.0
            self._percentiles[fpr] = float(np.percentile(scores, pct))
        # Store fine-grained CDF for arbitrary FPR interpolation
        n_pct = min(200, len(scores))
        pct_range = np.linspace(0.0, 100.0, n_pct)
        self._cdf_percentiles = pct_range.tolist()
        self._cdf_scores = np.percentile(scores, pct_range).tolist()

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to z-scores relative to normal distribution."""
        return (np.asarray(scores, dtype=np.float64) - self.normal_mean) / self.normal_std

    def threshold_at_fpr(self, target_fpr: float = 0.05) -> float:
        """Threshold corresponding to target FPR on the normal distribution.

        Uses pre-computed percentiles (no Gaussian assumption).
        target_fpr=0.05 → 95th percentile of normal scores → 5% normal traffic
        triggers a false alarm.
        """
        # Exact match
        if target_fpr in self._percentiles:
            return self._percentiles[target_fpr]
        # Interpolate from stored CDF
        if self._cdf_scores:
            target_pct = (1.0 - target_fpr) * 100.0
            target_pct = float(np.clip(target_pct, 0.0, 100.0))
            score = float(np.interp(target_pct, self._cdf_percentiles, self._cdf_scores))
            return score
        # Fallback if fit() was never called
        return self.normal_mean + self.normal_std * 1.645  # ~5% tail, approx only

    @property
    def percentile_95(self) -> float:
        return self._percentiles.get(0.05, self.normal_mean + 1.645 * self.normal_std)

    @property
    def percentile_99(self) -> float:
        return self._percentiles.get(0.01, self.normal_mean + 2.326 * self.normal_std)

    def to_dict(self) -> dict:
        return {
            "normal_mean": self.normal_mean,
            "normal_std": self.normal_std,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "percentiles": self._percentiles,
            # Store CDF for full fidelity reconstruction
            "cdf_scores": self._cdf_scores,
            "cdf_percentiles": self._cdf_percentiles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyScoreNormalizer":
        obj = cls()
        obj.normal_mean = d["normal_mean"]
        obj.normal_std = d["normal_std"]
        obj._percentiles = {float(k): float(v) for k, v in d.get("percentiles", {}).items()}
        obj._cdf_scores = d.get("cdf_scores", [])
        obj._cdf_percentiles = d.get("cdf_percentiles", [])
        # Backward compat: reconstruct percentiles from old format
        if not obj._percentiles:
            if "percentile_95" in d:
                obj._percentiles[0.05] = float(d["percentile_95"])
            if "percentile_99" in d:
                obj._percentiles[0.01] = float(d["percentile_99"])
        return obj


def export_onnx(model: nn.Module, save_path: str | Path, device: str = "cpu") -> None:
    """Export model to ONNX format with dynamic batch size."""
    model.eval()
    model.to(device)
    n_features = int(getattr(model, "n_features", 15))
    seq_len = int(getattr(model, "seq_len", 6))
    dummy_input = torch.zeros(1, seq_len, n_features, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        input_names=["input"],
        output_names=["reconstruction", "latent"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "reconstruction": {0: "batch_size"},
            "latent": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX exported: {save_path} (input: [batch, {seq_len}, {n_features}])")
