#!/usr/bin/env bash
set -euo pipefail

predict_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_path="${NETCATCH_PREDICT_CONFIG:-${predict_dir}/config.json}"

cd "${predict_dir}"
exec python3 -m netcatch_predict.node --config "${config_path}" "$@"
