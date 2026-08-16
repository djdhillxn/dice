#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [EPISODES] [NUM_ENVS] [OUTPUT_DIR]"
  exit 1
fi

CHECKPOINT="$1"
EPISODES="${2:-500}"
NUM_ENVS="${3:-256}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=$(cd "$(dirname "$CHECKPOINT")" && pwd)
EVAL_DIR="${4:-${RUN_DIR}/evaluation/${TIMESTAMP}}"

echo "[DICE] Running evaluation output to: ${EVAL_DIR}"

python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Eval-v0 \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --num_envs "$NUM_ENVS" \
  --seed 2026 \
  --headless \
  --output "${EVAL_DIR}/nominal"

python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Robust-v0 \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --num_envs "$NUM_ENVS" \
  --seed 2027 \
  --headless \
  --output "${EVAL_DIR}/robust"

python - <<PY
import json
from pathlib import Path

eval_dir = Path("${EVAL_DIR}")
nominal = json.loads((eval_dir / "nominal" / "summary.json").read_text())
robust = json.loads((eval_dir / "robust" / "summary.json").read_text())
output = {
    "project": "DICE",
    "evaluation_dir": str(eval_dir),
    "nominal": nominal,
    "robust": robust,
}
(eval_dir / "final_summary.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
PY
