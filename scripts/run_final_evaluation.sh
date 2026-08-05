#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?"Usage: scripts/run_final_evaluation.sh MODEL.zip VECNORMALIZE.pkl"}
VECNORMALIZE=${2:?"Usage: scripts/run_final_evaluation.sh MODEL.zip VECNORMALIZE.pkl"}
EPISODES=${EPISODES:-500}
NUM_ENVS=${NUM_ENVS:-256}
RUN_ROBUST=${RUN_ROBUST:-1}

for SEED in 2026 2027 2028; do
  python scripts/evaluate.py \
    --task DiceDial-Shadow-Sequence-v0 \
    --model "$MODEL" \
    --vecnormalize "$VECNORMALIZE" \
    --episodes "$EPISODES" \
    --num_envs "$NUM_ENVS" \
    --seed "$SEED" \
    --output "evaluation/nominal_seed_${SEED}" \
    --headless
done

python scripts/aggregate_evaluations.py \
  --inputs 'evaluation/nominal_seed_*/summary.json' \
  --output evaluation/nominal_aggregate

if [[ "$RUN_ROBUST" == "1" ]]; then
  for SEED in 3026 3027 3028; do
    python scripts/evaluate.py \
      --task DiceDial-Shadow-Robust-v0 \
      --model "$MODEL" \
      --vecnormalize "$VECNORMALIZE" \
      --episodes "$EPISODES" \
      --num_envs "$NUM_ENVS" \
      --seed "$SEED" \
      --output "evaluation/robust_seed_${SEED}" \
      --headless
  done

  python scripts/aggregate_evaluations.py \
    --inputs 'evaluation/robust_seed_*/summary.json' \
    --output evaluation/robust_aggregate
fi
