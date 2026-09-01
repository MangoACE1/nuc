#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.json"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="$1"
  shift
fi

if [[ ! -r "${CONFIG_PATH}" ]]; then
  echo "UDP config is not readable: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -r /opt/ros/humble/setup.bash ]]; then
  echo "ROS Humble setup not found: /opt/ros/humble/setup.bash" >&2
  exit 2
fi

set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m netcatch_udp.sender_node --config "${CONFIG_PATH}" "$@"
