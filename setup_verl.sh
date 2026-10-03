#!/usr/bin/env bash
set -euo pipefail

MATRIXREWARD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MATRIXREWARD_ROOT}"
if [[ ! -f third_party/verl/verl/trainer/main_ppo.py ]]; then
  echo "The bundled verl training source is missing." >&2
  exit 2
fi
"${PYTHON:-python3}" -m pip install -r requirements-verl.txt
