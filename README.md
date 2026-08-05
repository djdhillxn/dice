# DiceDial

**Command-conditioned in-hand die reorientation with an Isaac Lab Shadow Hand and RSL-RL PPO.**

A single policy receives a requested die face, rotates the die in-hand until that face points upward, holds it stable, and then receives another command without releasing the object. The rendered demonstration cycles through `1 → 6 → 3 → 5 → 2 → 4`.

## Why this implementation stays small

DiceDial extends Isaac Lab's existing direct Shadow Hand cube-reorientation environment. It does **not** rewrite the hand controller, contact physics, state extraction, vectorization, resets, or PPO. The custom code is limited to:

- face-number geometry
- command sampling and one-hot observations
- a yaw-invariant face-up reward
- hold-to-confirm success and command switching
- task metrics, evaluation, and video annotation
- a local numbered-die USD asset

## Repository layout

```text
source/dicedial/
  geometry.py                 # face normals and quaternion helpers
  callbacks.py                # SB3/TensorBoard task diagnostics
  tasks/
    dice_dial_env.py          # thin task subclass
    dice_dial_env_cfg.py      # Easy, Random, Sequence, Robust, Play configs
  agents/sb3_ppo_cfg.yaml     # PPO configuration
  assets/numbered_die.usda    # visual die + one cube collider
scripts/
  smoke_test.py
  train.py
  train_curriculum.sh
  evaluate.py
  aggregate_evaluations.py
  run_final_evaluation.sh
  play.py
  annotate_video.py
  plot_metrics.py
tests/test_geometry.py
docs/
```

## Supported stack

This repository targets **Isaac Lab / Isaac Sim / Python**. Use Ubuntu 22.04 or Windows 11 with a supported NVIDIA GPU. A GPU with at least 16 GB VRAM is the practical target for the default 2,048 parallel environments; lower `--num_envs` when necessary.

Isaac Lab is the successor to the standalone Isaac Gym stack. The project name retains the original idea, but the implementation uses the maintained Isaac Lab environment and wrappers.

## Installation

Create a clean environment and install the packages:

```bash
conda create -n dicedial python -y
conda activate dicedial

pip install -U torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install isaacsim \
  --extra-index-url https://pypi.nvidia.com
```

Install Isaac Lab:

```bash
git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..
```

Install DiceDial into the same Python environment:

```bash
cd DiceDial
pip install -e ".[video,test]"
```

Verify the pure geometry code without launching Isaac Sim:

```bash
pytest
```

Verify the full simulator task:

```bash
python scripts/smoke_test.py \
  --task DiceDial-Shadow-Random-v0 \
  --num_envs 16 \
  --steps 200 \
  --headless
```

The first Isaac run can pause while NVIDIA assets are downloaded and cached.

## Environments

| Environment | Purpose |
|---|---|
| `DiceDial-Shadow-Sequence-v0` | Main training — ACL starts with relaxed thresholds and tightens automatically |
| `DiceDial-Shadow-Robust-v0` | Evaluation-only; die mass/friction variation ±20 % |
| `DiceDial-Shadow-Play-v0` | One environment and deterministic six-face sequence for video |

The old `DiceDial-Shadow-Easy-v0` and `DiceDial-Shadow-Random-v0` stage
environments have been removed.  Their roles are absorbed by the ACL levels
(Level 0 and Level 1) inside the single training environment.

All environments expose a 20-dimensional continuous action and a **171-dimensional**
observation.  The observation is the inherited 157-dimensional full Shadow Hand
observation, extended with a six-way target one-hot, scalar hold progress, commanded
face normal in world frame (3), current top-face normal in world frame (3), and
alignment scalar (1).

## Training

### Single strong training run (recommended)

```bash
python scripts/train_rsl.py \
  --num_envs 2048 \
  --max_iterations 50000 \
  --run_name strong_run \
  --headless
```

This runs a single continuous RSL-RL PPO job on `DiceDial-Shadow-Sequence-v0`.
The **Automatic Curriculum Learning (ACL)** manager tightens the success thresholds
automatically as the policy improves — no manual stage transitions.

| ACL Level | `success_angle_deg` | `hold_steps` | Advance when… |
|---|---|---|---|
| 0 — relaxed | 30° | 8 | cmds/ep mean > 0.5 |
| 1 | 24° | 12 | cmds/ep mean > 1.0 |
| 2 | 20° | 16 | cmds/ep mean > 1.5 |
| 3 — final | 16° | 20 | — |

The training algorithm is **RSL-RL on-GPU PPO** with `num_steps_per_env=128`,
adaptive learning-rate schedule, and domain randomisation (mass ±20 %,
friction ±20 %) active from the first step.

Model checkpoints are saved at ACL advancement events and at the end:

```text
outputs/DiceDial-Shadow-Sequence-v0/strong_run/model_acl_Level_3_*_final.pt
outputs/DiceDial-Shadow-Sequence-v0/strong_run/model_final.pt
```

Monitor training:

```bash
tensorboard --logdir outputs
```

### Resume from a checkpoint

```bash
python scripts/train_rsl.py \
  --task DiceDial-Shadow-Sequence-v0 \
  --resume outputs/DiceDial-Shadow-Sequence-v0/strong_run/model_5000.pt \
  --max_iterations 50000 \
  --run_name strong_run_continued \
  --num_envs 2048 \
  --headless
```

The ACL state is automatically restored from the `.acl.json` file paired with
the checkpoint.

### SB3 fallback

The original SB3 PPO trainer (`scripts/train.py`) is still available as a
fallback.  Its `n_steps` has been raised to 128.  Evaluation and video
scripts (`evaluate.py`, `play.py`) continue to use SB3 for compatibility.

## Evaluation

Run deterministic evaluation on held-out reset randomness:

```bash
python scripts/evaluate.py \
  --task DiceDial-Shadow-Sequence-v0 \
  --model outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model.zip \
  --vecnormalize outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model_vecnormalize.pkl \
  --episodes 500 \
  --num_envs 256 \
  --seed 2026 \
  --output evaluation/seed_2026 \
  --headless
```

This writes:

- `summary.json`
- `episodes.csv`

The headline metrics are target-face success rate, median time to target, drop rate, and consecutive commands before failure.

For final reporting, evaluate three held-out seeds and report mean ± standard deviation. Do not select the seed with the best video as the quantitative result.

Run the same frozen policy under mild held-out die mass and friction variation:

```bash
python scripts/evaluate.py \
  --task DiceDial-Shadow-Robust-v0 \
  --model outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model.zip \
  --vecnormalize outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model_vecnormalize.pkl \
  --episodes 500 \
  --num_envs 256 \
  --seed 3026 \
  --output evaluation/robust_seed_3026 \
  --headless
```

Keep nominal and randomized results separate. The robustness task changes only object mass (0.8–1.2×) and friction (0.8–1.2), using Isaac Lab's built-in event functions.

Run the complete three-seed nominal and robustness protocol, followed by mean/standard-deviation aggregation:

```bash
scripts/run_final_evaluation.sh \
  outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model.zip \
  outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model_vecnormalize.pkl
```

Override `EPISODES`, `NUM_ENVS`, or `RUN_ROBUST` as environment variables for shorter diagnostic runs. Aggregated JSON and CSV files are written under `evaluation/*_aggregate/`.

## Render the command-sequence video

Rendering must run with cameras enabled and one environment:

```bash
python scripts/play.py \
  --task DiceDial-Shadow-Play-v0 \
  --model outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model.zip \
  --vecnormalize outputs/DiceDial-Shadow-Sequence-v0/stage3_sequence/model_vecnormalize.pkl \
  --output videos/final
```

The raw MP4 appears under `videos/final/raw/`, and frame-level task metadata is written to `videos/final/video_metrics.csv`.

Add the presentation overlay:

```bash
python scripts/annotate_video.py \
  --video videos/final/raw/dicedial-episode-0.mp4 \
  --metrics videos/final/video_metrics.csv \
  --output videos/final/dicedial_annotated.mp4
```

The overlay shows requested face, current top face, alignment, hold progress, and completed commands.

## Reward and success definition

The commanded face normal is rotated from the die frame into the world frame. Its dot product with world up is the principal alignment signal. Because the reward constrains only the selected face normal, yaw remains free; the policy does not have to match one arbitrary quaternion.

A command is complete only after the following remain true for consecutive control steps:

- requested face is within the configured angular tolerance of world up
- die remains near the palm
- die angular speed is below the settling threshold

The sequence task then samples a different face without resetting the hand or die.

## Troubleshooting

### Local USD fails to load

Verify that DiceDial was installed editable so package data resolves from the repository:

```bash
pip install -e ".[video,test]"
```

To isolate asset problems while preserving all task logic, temporarily use Isaac Lab's stock cube:

```bash
export DICEDIAL_USE_STOCK_CUBE=1
python scripts/smoke_test.py --num_envs 16 --headless
```

### Out of GPU memory

Reduce parallel environments:

```bash
python scripts/train.py --num_envs 512 --task DiceDial-Shadow-Easy-v0 --headless
```

Keep the PPO rollout batch compatible with the configured minibatch. Isaac Lab's SB3 processor also supports replacing `batch_size` with `n_minibatches` in the YAML when frequently changing environment count.

### Policy ignores the command

Inspect performance separately for all six target faces. Do not proceed to the sequence stage until every face receives successful examples. Increase Stage 2 training before changing the reward.

### Policy crosses the target but never succeeds

The settling gate is too strict for the current policy. First inspect angular speed and hold progress. For diagnosis, modestly raise `success_angular_speed` or reduce `hold_steps`; keep the final evaluation threshold fixed once chosen.

## Scope and claims

This is a simulation-only, privileged-state dexterous-manipulation project. It demonstrates command-conditioned in-hand reorientation and continuous command execution. It does not establish vision-based manipulation, general object dexterity, or sim-to-real transfer.
