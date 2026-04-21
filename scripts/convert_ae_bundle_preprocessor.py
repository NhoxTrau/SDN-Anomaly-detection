from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

from sdn_nids_realtime.train_v2.bundle_utils import export_quantile_uniform_standard_stats
from sdn_nids_realtime.train_v2.common import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AE runtime bundle preprocessing from sklearn pickle to JSON stats.")
    parser.add_argument("--bundle-path", required=True, help="Path to runtime_bundle.json or its bundle directory")
    parser.add_argument("--output-name", default="preprocessor_stats.json", help="Output stats filename inside the bundle directory")
    args = parser.parse_args()

    manifest_path = resolve_project_path(args.bundle_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "runtime_bundle.json"
    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocessing = dict(manifest.get("preprocessing", {}) or {})
    if str(preprocessing.get("kind", "")).strip() != "sklearn_pipeline":
        raise ValueError(f"Bundle preprocessing kind is not sklearn_pipeline: {preprocessing.get('kind')!r}")

    pkl_path = bundle_dir / str(preprocessing["preprocessor_filename"])
    with pkl_path.open("rb") as handle:
        preprocessor = pickle.load(handle)

    stats_path = bundle_dir / str(args.output_name)
    export_quantile_uniform_standard_stats(preprocessor, stats_path)

    backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    manifest["preprocessing"] = {
        "kind": "quantile_uniform_standard_stats",
        "preprocessor_stats_filename": stats_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "bundle_manifest": str(manifest_path),
        "backup_manifest": str(backup_path),
        "stats_path": str(stats_path),
        "preprocessing_kind": manifest["preprocessing"]["kind"],
    }, indent=2))


if __name__ == "__main__":
    main()
