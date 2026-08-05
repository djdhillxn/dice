#!/usr/bin/env bash
# ============================================================
# DEPRECATED — the 3-stage hard curriculum has been replaced.
# ============================================================
#
# Use the single-run RSL-RL training script instead:
#
#   python scripts/train_rsl.py \
#       --num_envs 2048 \
#       --max_iterations 50000 \
#       --run_name strong_run \
#       --headless
#
# The new training strategy:
#   - RSL-RL on-GPU PPO (5-10x faster than SB3)
#   - num_steps_per_env=128 for proper credit assignment
#   - Automatic Curriculum Learning (ACL) via rolling success-rate gating
#   - Stable-gated wrong-face penalty (no penalty during rotation)
#   - Domain randomization from day one (mass ±20%, friction ±20%)
#   - Face-normal supplementary observations (yaw-invariant signal)
#
# This file is kept only as a historical reference.
echo "ERROR: train_curriculum.sh is deprecated." >&2
echo "Run: python scripts/train_rsl.py --num_envs 2048 --max_iterations 50000 --headless" >&2
exit 1
