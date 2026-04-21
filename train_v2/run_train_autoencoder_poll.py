from __future__ import annotations

import argparse

from .common import DEFAULT_AE_PREPARED_DIR, DEFAULT_BUNDLE_DIR, DEFAULT_FEATURE_SCHEME, DEFAULT_OUTPUT_DIR
from .prepare_ae_poll_data import load_prepared_v2, prepare_ae_poll_data
from .train_ae_v2 import train_ae_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder on synthetic OpenFlow poll sequences built from InSDN.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--prepared-dir", default=str(DEFAULT_AE_PREPARED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--model-name", default="lstm_autoencoder_poll")
    parser.add_argument("--feature-scheme", default=DEFAULT_FEATURE_SCHEME, choices=["insdn_runtime10_v2", "insdn_ml_core_v1", "insdn_openflow_v1", "telemetry_v2", "telemetry_v1"])
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--max-polls", type=int, default=6)
    parser.add_argument("--key-mode", default="flow", choices=["service", "flow"])
    parser.add_argument("--source-prepared-dir", default="")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if args.force_prepare or args.dataset_dir is not None:
        dataset = prepare_ae_poll_data(
            dataset_dir=args.dataset_dir,
            save_dir=args.prepared_dir,
            source_prepared_dir=(args.source_prepared_dir or None),
            seq_len=args.seq_len,
            feature_scheme=args.feature_scheme,
            poll_interval_s=args.poll_interval_s,
            max_polls=args.max_polls,
            key_mode=args.key_mode,
        )
    else:
        dataset = load_prepared_v2(args.prepared_dir)
    seq_len = int(dataset.get("config", {}).get("seq_len", args.seq_len))
    train_ae_v2(
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
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        model_name=args.model_name,
        feature_scheme=str(dataset.get("config", {}).get("feature_scheme", args.feature_scheme)),
    )


if __name__ == "__main__":
    main()
