from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DEFAULT_FEATURE_SCHEME, feature_names_for_scheme


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.size(1)]


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * hidden_states).sum(dim=1)


class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 96,
        n_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 4,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.input_norm = nn.LayerNorm(n_features)
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.attn_pool = SelfAttentionPooling(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = FeedForward(hidden_dim, max(32, hidden_dim // 2), 1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(x)
        hidden, _ = self.encoder(x)
        hidden = self.proj(hidden)
        pooled = self.attn_pool(hidden)
        pooled = self.norm(pooled)
        logits = self.head(pooled).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


class TransformerSequenceClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 96,
        nhead: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        seq_len: int = 4,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Sequential(nn.Linear(n_features, d_model), nn.Dropout(dropout * 0.5))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.positional_encoding = LearnedPositionalEncoding(seq_len=seq_len + 1, d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = FeedForward(d_model, max(32, d_model // 2), 1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        x = self.input_norm(x)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        logits = self.head(cls_out).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return logits, probabilities

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[1]


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 4,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_norm = nn.LayerNorm(n_features)
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.enc_pool = SelfAttentionPooling(hidden_dim * 2)
        self.encoder_fc = nn.Sequential(nn.Linear(hidden_dim * 2, latent_dim), nn.Tanh())
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        hidden, _ = self.encoder_lstm(x)
        pooled = self.enc_pool(hidden)
        return self.encoder_fc(pooled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        base = self.decoder_fc(latent)
        repeated = base.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm(repeated)
        return self.output_fc(out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))


MODEL_REGISTRY: dict[str, tuple[type[nn.Module], str]] = {
    "lstm": (LSTMSequenceClassifier, "classifier"),
    "transformer": (TransformerSequenceClassifier, "classifier"),
    "autoencoder": (LSTMAutoencoder, "autoencoder"),
    "lstm_ae": (LSTMAutoencoder, "autoencoder"),
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type must be one of {list(MODEL_REGISTRY.keys())}")
    model_cls, _ = MODEL_REGISTRY[model_type]
    return model_cls(**kwargs)


def get_model_task(model_type: str) -> str:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type must be one of {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type][1]


def export_onnx(model: nn.Module, save_path: str, device: str = "cpu") -> None:
    model.eval()
    model.to(device)
    n_features = int(getattr(model, "n_features", len(feature_names_for_scheme(DEFAULT_FEATURE_SCHEME))))
    seq_len = int(getattr(model, "seq_len", 4))
    dummy_input = torch.zeros(1, seq_len, n_features, device=device)

    if hasattr(model, "predict_scores"):
        output_names = ["logits", "probabilities"]
        dynamic_axes = {"input": {0: "batch_size"}, "logits": {0: "batch_size"}, "probabilities": {0: "batch_size"}}
    else:
        output_names = ["reconstruction", "latent"]
        dynamic_axes = {"input": {0: "batch_size"}, "reconstruction": {0: "batch_size"}, "latent": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
