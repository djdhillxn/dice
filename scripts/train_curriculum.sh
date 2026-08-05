#!/usr/bin/env bash
set -euo pipefail

# Fast first-pass curriculum. Increase the three budgets for final training.
python scripts/train.py \
  --task DiceDial-Shadow-Easy-v0 \
  --run_name stage1_easy \
  --num_envs 2048 \
  --total_timesteps 5000000 \
  --headless

python scripts/train.py \
  --task DiceDial-Shadow-Random-v0 \
  --run_name stage2_random \
  --num_envs 2048 \
  --total_timesteps 10000000 \
  --checkpoint outputs/DiceDial-Shadow-Easy-v0/stage1_easy/model.zip \
  --vecnormalize outputs/DiceDial-Shadow-Easy-v0/stage1_easy/model_vecnormalize.pkl \
  --headless

python scripts/train.py \
  --task DiceDial-Shadow-Sequence-v0 \
  --run_name stage3_sequence \
  --num_envs 2048 \
  --total_timesteps 20000000 \
  --checkpoint outputs/DiceDial-Shadow-Random-v0/stage2_random/model.zip \
  --vecnormalize outputs/DiceDial-Shadow-Random-v0/stage2_random/model_vecnormalize.pkl \
  --headless
