from __future__ import annotations

import time


def _bg(host, cmd: str, log_name: str) -> None:
    host.cmd(f"{cmd} > /tmp/{log_name} 2>&1 &")


def _start_http_server(host, port: int, log_name: str) -> None:
    host.cmd(f"pkill -f 'python3 -m http.server {port}' >/dev/null 2>&1")
    host.cmd(f"python3 -m http.server {port} >/tmp/{log_name} 2>&1 &")


def _start_echo_service(host, port: int, log_name: str) -> None:
    host.cmd("pkill -f 'sdn_nids_echo_service' >/dev/null 2>&1")
    script = (
        "python3 -c \"# sdn_nids_echo_service\n"
        "import socketserver\n"
        f"PORT={int(port)}\n"
        "class H(socketserver.BaseRequestHandler):\n"
        "    def handle(self):\n"
        "        data = self.request.recv(4096)\n"
        "        if data:\n"
        "            self.request.sendall(b'ok\\n' + data[:64])\n"
        "socketserver.ThreadingTCPServer.allow_reuse_address = True\n"
        "srv = socketserver.ThreadingTCPServer(('0.0.0.0', PORT), H)\n"
        "srv.serve_forever()\""
    )
    host.cmd(f"{script} >/tmp/{log_name} 2>&1 &")


def setup_services(net) -> None:
    h1 = net.get("h1")
    h2 = net.get("h2")

    h1.cmd("pkill -f 'iperf3 -s' >/dev/null 2>&1")
    _start_http_server(h1, 80, "h1_http.log")
    h1.cmd("iperf3 -s -D --logfile /tmp/h1_iperf.log")

    h2.cmd("pkill -f 'nc -lk 22' >/dev/null 2>&1")
    h2.cmd("pkill -f 'iperf3 -s -p 5202' >/dev/null 2>&1")
    h2.cmd("nc -lk 22 >/tmp/h2_nc.log 2>&1 &")
    h2.cmd("iperf3 -s -p 5202 -D --logfile /tmp/h2_iperf.log")
    _start_http_server(h2, 8081, "h2_c2_http.log")
    _start_echo_service(h2, 6200, "h2_shellsim.log")
    print("Started runtime demo services on h1/h2")


def run_ping_only(net, duration: int = 20, target_ip: str = "10.0.0.1") -> None:
    h3 = net.get("h3")
    print(f"Running ping_only for {duration}s")
    h3.cmd(f"ping -i 0.5 -c {max(2, duration * 2)} {target_ip} >/tmp/h3_ping_only.log 2>&1 &")
    time.sleep(duration)


def run_http_single(net, duration: int = 20, target_ip: str = "10.0.0.1") -> None:
    h3 = net.get("h3")
    print(f"Running http_single for {duration}s")
    _bg(
        h3,
        f"bash -lc 'for i in $(seq 1 {max(4, duration // 2)}); do curl -m 2 -s http://{target_ip} >/dev/null; sleep 2; done'",
        "h3_http_single.log",
    )
    time.sleep(duration)


def run_iperf_single(net, duration: int = 20, target_ip: str = "10.0.0.1") -> None:
    h4 = net.get("h4")
    print(f"Running iperf_single for {duration}s")
    h4.cmd(f"iperf3 -c {target_ip} -t {duration} -b 5M >/tmp/h4_iperf_single.log 2>&1 &")
    time.sleep(duration)


def run_benign(net, duration: int = 30) -> None:
    h3 = net.get("h3")
    h4 = net.get("h4")
    print(f"Running benign traffic for {duration}s")
    h3.cmd(f"ping -i 0.5 -c {duration * 2} 10.0.0.1 >/tmp/h3_ping.log 2>&1 &")
    h3.cmd(
        f"for i in $(seq 1 {max(1, duration // 3)}); do "
        f"curl -m 3 -s http://10.0.0.1 >/dev/null; sleep 3; done >/tmp/h3_http.log 2>&1 &"
    )
    h4.cmd(f"iperf3 -c 10.0.0.1 -t {duration} -b 10M >/tmp/h4_iperf.log 2>&1 &")
    time.sleep(duration)


def run_syn(net, duration: int = 20, target_ip: str = "10.0.0.1", target_port: int = 80) -> None:
    h7 = net.get("h7")
    print(f"Running SYN flood for {duration}s")
    h7.cmd(f"timeout {duration} hping3 -S --flood -p {target_port} {target_ip} >/tmp/h7_syn.log 2>&1 &")
    time.sleep(duration + 1)
    h7.cmd("killall hping3 >/dev/null 2>&1")


def run_udp(net, duration: int = 20, target_ip: str = "10.0.0.1", target_port: int = 53) -> None:
    h5 = net.get("h5")
    print(f"Running UDP flood for {duration}s")
    h5.cmd(f"timeout {duration} hping3 --udp --flood -p {target_port} {target_ip} >/tmp/h5_udp.log 2>&1 &")
    time.sleep(duration + 1)
    h5.cmd("killall hping3 >/dev/null 2>&1")


def run_probe(net, duration: int = 18, target_ip: str = "10.0.0.1") -> None:
    h6 = net.get("h6")
    print(f"Running probe / scan for {duration}s")
    _bg(
        h6,
        f"bash -lc 'end=$((SECONDS+{duration})); while [ $SECONDS -lt $end ]; do nmap -Pn -sS -T4 --max-retries 0 --min-rate 400 -p 1-256 {target_ip}; sleep 1; nmap -Pn -sS -T4 --max-retries 0 --min-rate 400 -p 257-512 {target_ip}; sleep 1; done'",
        "h6_probe.log",
    )
    time.sleep(duration)


def run_bfa_ssh(net, duration: int = 25, target_ip: str = "10.0.0.2", target_port: int = 22) -> None:
    h6 = net.get("h6")
    h5 = net.get("h5")
    print(f"Running bfa_ssh traffic-shape simulation for {duration}s")
    for host, log_name in ((h6, "h6_bfa_ssh.log"), (h5, "h5_bfa_ssh.log")):
        _bg(
            host,
            f"bash -lc 'end=$((SECONDS+{duration})); while [ $SECONDS -lt $end ]; do for i in $(seq 1 10); do timeout 1 nc -z -w 1 {target_ip} {target_port}; done; sleep 0.5; done'",
            log_name,
        )
    time.sleep(duration)


def run_botnet_beacon(net, duration: int = 60, c2_ip: str = "10.0.0.2", c2_port: int = 8081) -> None:
    print(f"Running botnet_beacon for {duration}s")
    for idx, host_name in enumerate(("h5", "h6", "h7"), start=1):
        host = net.get(host_name)
        _bg(
            host,
            f"bash -lc 'end=$((SECONDS+{duration})); while [ $SECONDS -lt $end ]; do curl -m 1 -s http://{c2_ip}:{c2_port}/beacon/{host_name} >/dev/null; sleep $((5 + ({idx} % 2))); done'",
            f"{host_name}_botnet_beacon.log",
        )
    time.sleep(duration)


def run_u2r_shape(net, duration: int = 35, target_ip: str = "10.0.0.2") -> None:
    h7 = net.get("h7")
    print(f"Running u2r_shape simulation for {duration}s")
    stage1 = (
        "python3 - <<'EOF'\n"
        "import time, urllib.request\n"
        f"target = 'http://{target_ip}:8081/trigger?payload=' + ('A' * 700)\n"
        "for _ in range(18):\n"
        "    try:\n"
        "        urllib.request.urlopen(target, timeout=1).read(1)\n"
        "    except Exception:\n"
        "        pass\n"
        "    time.sleep(0.2)\n"
        "EOF"
    )
    _bg(h7, f"bash -lc \"{stage1}\"", "h7_u2r_stage1.log")
    _bg(
        h7,
        f"bash -lc 'sleep 5; end=$((SECONDS+{max(8, duration - 6)})); while [ $SECONDS -lt $end ]; do printf \"id\\nuname -a\\nwhoami\\n\" | nc -w 1 {target_ip} 6200 >/dev/null; sleep 1; done'",
        "h7_u2r_stage2.log",
    )
    time.sleep(duration)


def run_benign_async(net, duration: int = 30) -> None:
    h3 = net.get("h3")
    h4 = net.get("h4")
    h3.cmd(f"ping -i 0.5 -c {duration * 2} 10.0.0.1 >/tmp/h3_ping.log 2>&1 &")
    h3.cmd(
        f"for i in $(seq 1 {max(1, duration // 3)}); do "
        f"curl -m 3 -s http://10.0.0.1 >/dev/null; sleep 3; done >/tmp/h3_http.log 2>&1 &"
    )
    h4.cmd(f"iperf3 -c 10.0.0.1 -t {duration} -b 10M >/tmp/h4_iperf.log 2>&1 &")


def run_syn_async(net, duration: int = 20, target_ip: str = "10.0.0.1", target_port: int = 80) -> None:
    net.get("h7").cmd(f"timeout {duration} hping3 -S --flood -p {target_port} {target_ip} >/tmp/h7_syn.log 2>&1 &")


def run_udp_async(net, duration: int = 20, target_ip: str = "10.0.0.1", target_port: int = 53) -> None:
    net.get("h5").cmd(f"timeout {duration} hping3 --udp --flood -p {target_port} {target_ip} >/tmp/h5_udp.log 2>&1 &")


def run_probe_async(net, target_ip: str = "10.0.0.1") -> None:
    net.get("h6").cmd(f"nmap -sS -T4 -p 1-1024 {target_ip} >/tmp/h6_probe.log 2>&1 &")


def run_mixed(net, benign_duration: int = 60, attack_duration: int = 20) -> None:
    print("Running mixed scenario")
    run_benign_async(net, benign_duration)
    time.sleep(10)
    run_syn_async(net, attack_duration)
    time.sleep(attack_duration)
    time.sleep(5)
    run_udp_async(net, attack_duration)
    time.sleep(5)
    run_probe_async(net)
    time.sleep(max(5, benign_duration - 40))


def run_demo_syn(net, warmup: int = 20, attack_duration: int = 15, recovery: int = 10) -> None:
    print("Running demo_syn: warmup -> attack -> recovery")
    run_benign_async(net, warmup + attack_duration + recovery)
    time.sleep(warmup)
    run_syn_async(net, attack_duration)
    time.sleep(attack_duration)
    time.sleep(recovery)


def run_demo_udp(net, warmup: int = 20, attack_duration: int = 15, recovery: int = 10) -> None:
    print("Running demo_udp: warmup -> attack -> recovery")
    run_benign_async(net, warmup + attack_duration + recovery)
    time.sleep(warmup)
    run_udp_async(net, attack_duration)
    time.sleep(attack_duration)
    time.sleep(recovery)


def run_benign_v2(net, duration: int = 60) -> None:
    """Benign scenario ổn định hơn, tránh chạm các pattern scan/BFA để làm baseline sạch."""
    h3 = net.get("h3")
    h4 = net.get("h4")
    h5 = net.get("h5")
    h6 = net.get("h6")
    h7 = net.get("h7")

    print(f"Running benign_v2 traffic for {duration}s")

    _bg(h3, f"bash -lc 'for i in $(seq 1 {max(8, duration // 2)}); do ping -c 1 -W 1 10.0.0.1 >/dev/null; sleep $((1 + RANDOM % 2)); done'", "h3_benign_v2_ping.log")
    _bg(h3, f"bash -lc 'for i in $(seq 1 {max(6, duration // 4)}); do curl -m 2 -s http://10.0.0.1 >/dev/null; sleep $((2 + RANDOM % 3)); done'", "h3_benign_v2_http.log")
    _bg(h4, f"bash -lc 'for i in $(seq 1 {max(5, duration // 4)}); do ping -c 1 -W 1 10.0.0.2 >/dev/null; sleep $((1 + RANDOM % 3)); done'", "h4_benign_v2_ping.log")
    _bg(h4, f"bash -lc 'for i in $(seq 1 {max(4, duration // 5)}); do curl -m 2 -s http://10.0.0.2:8081 >/dev/null; sleep $((2 + RANDOM % 4)); done'", "h4_benign_v2_http.log")
    h5.cmd(f"iperf3 -c 10.0.0.1 -u -b 500K -t {max(12, duration // 2)} >/tmp/h5_benign_v2_iperf.log 2>&1 &")
    h7.cmd(f"iperf3 -c 10.0.0.2 -u -p 5202 -b 500K -t {max(12, duration // 2)} >/tmp/h7_benign_v2_iperf.log 2>&1 &")
    _bg(h6, f"bash -lc 'for i in $(seq 1 {max(4, duration // 6)}); do curl -m 2 -s http://10.0.0.1 >/dev/null; sleep $((3 + RANDOM % 4)); done'", "h6_benign_v2_http.log")
    _bg(h7, f"bash -lc 'for i in $(seq 1 {max(5, duration // 5)}); do ping -c 1 -W 1 10.0.0.1 >/dev/null; sleep $((2 + RANDOM % 3)); done'", "h7_benign_v2_ping.log")
    time.sleep(duration)


SCENARIOS = {
    "ping_only": run_ping_only,
    "http_single": run_http_single,
    "iperf_single": run_iperf_single,
    "benign": run_benign,
    "benign_v2": run_benign_v2,
    "syn": run_syn,
    "udp": run_udp,
    "probe": run_probe,
    "bfa_ssh": run_bfa_ssh,
    "botnet_beacon": run_botnet_beacon,
    "u2r_shape": run_u2r_shape,
    "mixed": run_mixed,
    "demo_syn": run_demo_syn,
    "demo_udp": run_demo_udp,
}


def cleanup_all(net) -> None:
    for host in net.hosts:
        host.cmd("killall hping3 nmap iperf3 ping curl nc >/dev/null 2>&1")
        host.cmd("pkill -f 'python3 -m http.server 80' >/dev/null 2>&1")
        host.cmd("pkill -f 'python3 -m http.server 8081' >/dev/null 2>&1")
        host.cmd("pkill -f 'sdn_nids_echo_service' >/dev/null 2>&1")
    print("Stopped all traffic generators and demo services")


def run_scenario(net, scenario_name: str, **kwargs) -> None:
    if scenario_name == "all":
        for name in ("ping_only", "http_single", "iperf_single", "benign_v2", "probe", "bfa_ssh", "botnet_beacon", "u2r_shape", "syn", "udp"):
            print(f"\n=== {name} ===")
            setup_services(net)
            SCENARIOS[name](net, **kwargs)
            cleanup_all(net)
            time.sleep(5)
        return

    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {list(SCENARIOS.keys()) + ['all']}")
        return

    setup_services(net)
    SCENARIOS[scenario_name](net, **kwargs)
    cleanup_all(net)


if __name__ == "__main__":
    print("Use inside Mininet CLI:")
    print("  mininet> py from sdn_nids_realtime.demo.scenarios import run_scenario")
    print("  mininet> py run_scenario(net, 'ping_only')")
    print("  mininet> py run_scenario(net, 'bfa_ssh')")
    print("  mininet> py run_scenario(net, 'botnet_beacon')")
    print("  mininet> py run_scenario(net, 'u2r_shape')")
