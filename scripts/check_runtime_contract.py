#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNTIME_SAFE = {
    'packet_count', 'byte_count', 'flow_duration_s', 'packet_rate', 'byte_rate',
    'avg_packet_size', 'dst_port_norm', 'protocol_tcp', 'protocol_udp', 'protocol_icmp'
}
LEGACY_ONLY = {'src_port_norm'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect bundle/runtime feature contract and flag train/runtime mismatches.')
    parser.add_argument('--bundle-path', required=True)
    args = parser.parse_args()
    path = Path(args.bundle_path)
    manifest_path = path / 'runtime_bundle.json' if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    features = list(manifest.get('feature_names', []))
    report = {
        'bundle': str(manifest_path),
        'feature_scheme': manifest.get('feature_scheme'),
        'n_features': len(features),
        'feature_names': features,
        'uses_legacy_src_port_norm': 'src_port_norm' in features,
        'runtime_safe_subset_ok': set(features).issubset(RUNTIME_SAFE | LEGACY_ONLY),
        'unsupported_runtime_features': sorted(set(features) - (RUNTIME_SAFE | LEGACY_ONLY)),
        'recommendation': (
            'Retrain with insdn_runtime10_v2 and recalibrate runtime threshold.'
            if 'src_port_norm' in features else
            'Feature contract already matches runtime-safe set.'
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
