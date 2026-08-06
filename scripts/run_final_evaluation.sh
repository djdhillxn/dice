#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [EPISODES] [NUM_ENVS]"
  exit 1
fi

CHECKPOINT="$1"
EPISODES="${2:-500}"
NUM_ENVS="${3:-256}"

python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Eval-v0 \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --num_envs "$NUM_ENVS" \
  --seed 2026 \
  --output evaluation/nominal

python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Robust-v0 \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --num_envs "$NUM_ENVS" \
  --seed 2027 \
  --output evaluation/robust

python - <<'PY'
import json
from pathlib import Path

nominal = json.loads(Path("evaluation/nominal/summary.json").read_text())
robust = json.loads(Path("evaluation/robust/summary.json").read_text())
output = {
    "project": "DICE",
    "nominal": nominal,
    "robust": robust,
}
Path("evaluation/final_summary.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
PY
