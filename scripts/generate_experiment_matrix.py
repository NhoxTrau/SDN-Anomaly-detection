from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a repeatable experiment matrix for the SDN realtime thesis.")
    parser.add_argument("--controller-ip", default="192.168.1.10")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--hosts", default="4,7,10")
    parser.add_argument("--poll-intervals", default="1.0,1.5,2.0")
    parser.add_argument("--scenarios", default="ping_only,benign_v2,probe,syn,udp")
    parser.add_argument("--output", default="artifacts_v4/benchmark/experiment_matrix.md")
    args = parser.parse_args()

    hosts = [int(x.strip()) for x in args.hosts.split(",") if x.strip()]
    poll_intervals = [float(x.strip()) for x in args.poll_intervals.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment Matrix",
        "",
        "This matrix is intended for the thesis workflow: polling thread -> feature thread -> inference thread, with scalability measured by host count and polling interval.",
        "",
        "## Matrix",
        "",
        "| Run ID | Hosts | Poll interval (s) | Scenario | Purpose |",
        "|---|---:|---:|---|---|",
    ]

    for host_count in hosts:
        for poll in poll_intervals:
            for scenario in scenarios:
                run_id = f"h{host_count}_p{str(poll).replace('.', 'p')}_{scenario}"
                purpose = "baseline" if scenario in {"ping_only", "benign_v2"} else "attack"
                lines.append(f"| `{run_id}` | {host_count} | {poll:.2f} | `{scenario}` | {purpose} |")

    lines.extend([
        "",
        "## Controller / dashboard terminal",
        "",
        "```bash",
        "export SDN_NIDS_RUNTIME_KEY_MODE=flow",
        "export SDN_NIDS_FLOW_MATCH_MODE=conversation",
        "export SDN_NIDS_ADAPTIVE_POLLING=0",
        "export SDN_NIDS_FLOW_IDLE_TIMEOUT=5",
        "export SDN_NIDS_POLL_REPLY_TIMEOUT=4.0",
        "export SDN_NIDS_INFERENCE_BATCH_MAX=512",
        "export SDN_NIDS_INFERENCE_BATCH_WAIT_MS=1.5",
        "export SDN_NIDS_POLL_BACKPRESSURE_THRESHOLD=8000",
        "export SDN_NIDS_HIGH_PRESSURE_THRESHOLD=5000",
        "export SDN_NIDS_TELEMETRY_SKIP_QUEUE_THRESHOLD=2500",
        "export SDN_NIDS_QUEUE_BACKPRESSURE_DROP_THRESHOLD=7000",
        "```",
        "",
        "For each run, additionally set:",
        "",
        "```bash",
        "export SDN_NIDS_RUN_ID=<RUN_ID>",
        "export SDN_NIDS_POLL_INTERVAL=<POLL_INTERVAL>",
        "bash scripts/run_dashboard.sh",
        "bash scripts/run_realtime_controller.sh",
        "```",
        "",
        "## Mininet terminal",
        "",
        "```bash",
        f"sudo -E python3 -m sdn_nids_realtime.demo.topology --hosts <HOSTS> --controller-ip {args.controller_ip} --controller-port {args.controller_port}",
        "```",
        "",
        "## Aggregating the results",
        "",
        "After all runs are complete, aggregate them into a thesis-friendly report:",
        "",
        "```bash",
        "python scripts/benchmark_scalability.py --runtime-root runtime_logs",
        "```",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote experiment matrix to {output}")


if __name__ == "__main__":
    main()
