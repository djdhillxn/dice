#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo "Usage: $0 CHECKPOINT [EPISODES] [NUM_ENVS] [OUTPUT_DIR] [--force]"
  exit 1
fi

CHECKPOINT=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
EPISODES="${2:-1000}"
NUM_ENVS="${3:-256}"
OUTPUT_DIR="${4:-}"
FORCE="${5:-}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${OUTPUT_DIR}" == "--force" && -z "${FORCE}" ]]; then
  FORCE="--force"
  OUTPUT_DIR=""
fi
if [[ -n "${FORCE}" && "${FORCE}" != "--force" ]]; then
  echo "The fifth argument must be --force when provided."
  exit 1
fi

cd "${REPO_ROOT}"

COMMAND=(
  python -u scripts/run_final_evaluation.py "${CHECKPOINT}"
  --episodes "${EPISODES}"
  --num-envs "${NUM_ENVS}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMAND+=(--output "${OUTPUT_DIR}")
fi
if [[ "${FORCE}" == "--force" ]]; then
  COMMAND+=(--force)
fi

"${COMMAND[@]}"
