#!/usr/bin/env bash
set -euo pipefail

nuc_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${NETCATCH_PYTHON:-python3}"

"${python_bin}" "${nuc_dir}/verify_contract.py"

mapfile -d '' python_modules < <(
  find \
    "${nuc_dir}/predict/netcatch_predict" \
    "${nuc_dir}/udp/netcatch_udp" \
    -type f -name '*.py' -print0
)
"${python_bin}" -m py_compile "${python_modules[@]}"

bash -n \
  "${nuc_dir}/predict/run_predict.sh" \
  "${nuc_dir}/udp/run_udp_sender.sh" \
  "${nuc_dir}/udp/run_udp_receiver.sh"

echo "NUC package verification passed. ROS/VRPN/multicast live smoke remains mandatory before flight."
