#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
NUM_ENVS="${1:-64}"
ITERATIONS="${2:-2}"

cd "${REPO_ROOT}"

echo "[DICE PREFLIGHT] Validating reset, actor/critic observations, wrapper, storage, and PPO."
echo "[DICE PREFLIGHT] Environments: ${NUM_ENVS}; iterations: ${ITERATIONS}"

python -u scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${ITERATIONS}" \
  --save_interval 1000 \
  --output_root outputs/preflight \
  --run_name asymmetric_contract \
  --headless

echo "[DICE PREFLIGHT] Complete. Inspect outputs/preflight/<latest-run>/run.json and startup.log."
