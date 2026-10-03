#!/usr/bin/env bash
set -euo pipefail

export MATRIXREWARD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${MATRIXREWARD_ROOT}:${MATRIXREWARD_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"
cd "${MATRIXREWARD_ROOT}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the supplied verl parquet file.}"

"${PYTHON:-python3}" - <<'PY'
from pathlib import Path
import os
import verl
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
expected = Path(os.environ["MATRIXREWARD_ROOT"]) / "third_party" / "verl" / "verl"
if Path(verl.__file__).resolve().parent != expected.resolve():
    raise SystemExit("Use the bundled verl source from third_party/verl.")
if ("run_batch" not in RewardLoopWorker.compute_score_batch.__code__.co_names
        or not hasattr(RayPPOTrainer, "_uses_grouped_reward")):
    raise SystemExit("The bundled verl group reward hooks are missing.")
PY

if [[ "${1:-}" == "--check" ]]; then
  shift
  exec "${PYTHON:-python3}" -m verl.trainer.main_ppo \
    --config-path="${MATRIXREWARD_ROOT}/configs" --config-name=verl --cfg job --resolve "$@"
fi
: "${JUDGE_API_URL:?Set JUDGE_API_URL to the judge chat-completions endpoint.}"
exec "${PYTHON:-python3}" -m verl.trainer.main_ppo \
  --config-path="${MATRIXREWARD_ROOT}/configs" --config-name=verl "$@"
