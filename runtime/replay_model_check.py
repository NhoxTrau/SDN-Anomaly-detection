from __future__ import annotations

import argparse
import json

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

from ..train_v2.prepare_data import load_prepared
from ..train_v2.common import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay prepared split through runtime bundle for sanity check.")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if ort is None:
        raise ImportError("onnxruntime is required")
    prepared = load_prepared(args.prepared_dir)
    manifest_path = resolve_project_path(args.bundle_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "runtime_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(manifest_path.parent / manifest["onnx_filename"]), providers=["CPUExecutionProvider"])
    X = np.asarray(prepared[f"X_{args.split}"], dtype=np.float32)
    y = np.asarray(prepared[f"y_binary_{args.split}"], dtype=np.int64)
    input_name = session.get_inputs()[0].name
    scores = []
    for start in range(0, len(X), args.batch_size):
        batch = X[start:start + args.batch_size]
        outs = session.run(None, {input_name: batch})
        if manifest["task_type"] == "classifier":
            score = np.asarray(outs[1] if len(outs) >= 2 else 1.0 / (1.0 + np.exp(-np.asarray(outs[0]))), dtype=np.float32).reshape(-1)
        else:
            recon = np.asarray(outs[0], dtype=np.float32)
            score = np.mean((recon - batch) ** 2, axis=(1, 2))
        scores.append(score)
    scores = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)
    print(json.dumps({
        "split": args.split,
        "n_samples": int(len(X)),
        "normal_mean": float(np.mean(scores[y == 0])) if np.any(y == 0) else None,
        "attack_mean": float(np.mean(scores[y == 1])) if np.any(y == 1) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
