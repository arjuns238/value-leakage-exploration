#!/usr/bin/env bash
# Runs on your LAPTOP. Opens the tunnel: localhost:8888 -> pod's 127.0.0.1:8888
#
#   ./connect.sh <host> <port>
#
# RunPod: take the host/port from Connect -> "SSH over exposed TCP"
#         (NOT the ssh.runpod.io proxy command - that can't forward ports)
set -euo pipefail

HOST="${1:?usage: ./connect.sh <host> <port>}"
PORT="${2:?usage: ./connect.sh <host> <port>}"

echo "tunneling localhost:8888 -> ${HOST}:${PORT}"
exec ssh -N \
  -L 8888:localhost:8888 \
  -p "$PORT" "root@${HOST}" \
  -i ~/.ssh/id_ed25519 \
  -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes
