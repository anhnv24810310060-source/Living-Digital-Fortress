#!/usr/bin/env bash
set -euo pipefail

# Usage: WG_IFACE=wg-shieldx WG_ADDR=10.10.0.1/24 WG_PORT=51820 ./scripts/wg-route-setup.sh

WG_IFACE=${WG_IFACE:-wg-shieldx}
WG_ADDR=${WG_ADDR:-10.10.0.1/24}
WG_PORT=${WG_PORT:-51820}

sudo ip link add dev "$WG_IFACE" type wireguard || true
sudo ip address add "$WG_ADDR" dev "$WG_IFACE" || true
sudo ip link set up dev "$WG_IFACE"

# Enable IP forwarding
sudo sysctl -w net.ipv4.ip_forward=1

# nftables NAT masquerade for egress (adjust table/chain to your env)
sudo nft add table ip nat || true
sudo nft add chain ip nat POSTROUTING '{ type nat hook postrouting priority 100 ; }' || true
# NAT egress for traffic coming from WireGuard subnet going out to Internet
sudo nft add rule ip nat POSTROUTING ip saddr $WG_ADDR oifname != "$WG_IFACE" counter masquerade || true

# Allow forwarding between $WG_IFACE and default egress
EGRESS_IF=$(ip route show default | awk '/default/ {print $5; exit}')
sudo nft add table ip filter || true
sudo nft add chain ip filter FORWARD '{ type filter hook forward priority 0 ; }' || true
sudo nft add rule ip filter FORWARD iifname "$WG_IFACE" oifname $EGRESS_IF ct state new,established,related accept || true
sudo nft add rule ip filter FORWARD iifname $EGRESS_IF oifname "$WG_IFACE" ct state established,related accept || true

echo "WireGuard interface $WG_IFACE up with $WG_ADDR. Forwarding enabled and NAT configured."


