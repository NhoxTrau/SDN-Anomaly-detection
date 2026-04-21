from __future__ import annotations

import argparse
import os
import time

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController

from .scenarios import setup_services


def start_services(net: Mininet) -> None:
    info("\n*** Disabling offload features for cleaner telemetry...\n")
    for host in net.hosts:
        intf_name = host.defaultIntf().name
        host.cmd(f"ethtool -K {intf_name} tx off rx off sg off tso off gso off gro off > /dev/null 2>&1")
    for switch in net.switches:
        for intf in switch.intfList():
            if intf.name != "lo":
                switch.cmd(f"ethtool -K {intf.name} tx off rx off sg off tso off gso off gro off > /dev/null 2>&1")
    info("*** Offloading disabled.\n")
    setup_services(net)


def build_topology(
    num_edge_hosts: int = 7,
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6653,
    link_bw: int = 100,
) -> Mininet:
    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
    )
    c0 = net.addController("c0", controller=RemoteController, ip=controller_ip, port=controller_port)
    s1 = net.addSwitch("s1")
    s2 = net.addSwitch("s2")
    s3 = net.addSwitch("s3")

    hosts = [net.addHost(f"h{i}", ip=f"10.0.0.{i}/24") for i in range(1, num_edge_hosts + 1)]
    net.addLink(s1, s2, bw=link_bw)
    net.addLink(s1, s3, bw=link_bw)
    for i, host in enumerate(hosts, start=1):
        if i <= max(3, num_edge_hosts // 2):
            net.addLink(s2, host, bw=link_bw)
        else:
            net.addLink(s3, host, bw=link_bw)

    net.build()
    c0.start()
    s1.start([c0])
    s2.start([c0])
    s3.start([c0])
    time.sleep(1)
    return net


def run_topology(
    num_edge_hosts: int = 7,
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6653,
    link_bw: int = 100,
) -> None:
    setLogLevel("info")
    info("*** Creating SDN testbed\n")
    info(f"*** Remote controller: {controller_ip}:{controller_port}\n")
    net = build_topology(
        num_edge_hosts=num_edge_hosts,
        controller_ip=controller_ip,
        controller_port=controller_port,
        link_bw=link_bw,
    )
    start_services(net)
    CLI(net)
    net.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mininet topology for telemetry-based SDN NIDS.")
    parser.add_argument("--hosts", type=int, default=7)
    parser.add_argument("--controller-ip", default=os.environ.get("SDN_NIDS_CONTROLLER_IP", "127.0.0.1"))
    parser.add_argument("--controller-port", type=int, default=int(os.environ.get("SDN_NIDS_CONTROLLER_PORT", "6653")))
    parser.add_argument("--link-bw", type=int, default=int(os.environ.get("SDN_NIDS_LINK_BW", "100")))
    args = parser.parse_args()
    run_topology(
        num_edge_hosts=args.hosts,
        controller_ip=args.controller_ip,
        controller_port=args.controller_port,
        link_bw=args.link_bw,
    )


if __name__ == "__main__":
    main()
