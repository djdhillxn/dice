# DICE

**DICE** trains a Shadow Hand to reorient a held die to a requested numbered face and then continue through new commands without releasing the object.

The repository intentionally has one main training path:

- Isaac Lab's existing direct Shadow Hand environment
- Isaac Lab's stock instanceable DexCube during training
- RSL-RL PPO using the official Shadow Hand network and optimizer defaults
- the complete final task from the first transition
- no curriculum and no alternate training path

## Task

At each command, the policy receives a requested face from 1 through 6. A command succeeds when all of the following remain true for 20 consecutive control steps:

- requested face is within **16 degrees** of world up
- cube remains within `0.12 m` of the in-hand reference position
- cube angular speed is at most `1.25 rad/s`

After success, a different face is selected immediately without resetting the hand or cube.

## Environments

| Environment | Purpose | Object | Randomization |
|---|---|---|---|
| `DICE-Shadow-Train-v0` | Main PPO training | stock instanceable DexCube | none |
| `DICE-Shadow-Eval-v0` | Nominal evaluation | stock instanceable DexCube | none |
| `DICE-Shadow-Robust-v0` | Held-out robustness evaluation | stock instanceable DexCube | mass and friction ±20% |
| `DICE-Shadow-Play-v0` | Six-command video | local numbered die | none |

## Observation and action spaces

The action is Isaac Lab's inherited 20-dimensional Shadow Hand joint-target action.

The policy observation has 165 dimensions:

- stock Shadow Hand full observation: 157
- requested-face one-hot: 6
- hold progress: 1
- requested-face alignment with world up: 1

Alignment is computed directly from the current cube quaternion and current command whenever observations are constructed. It is not copied from a reward-side cache.

## Reward

The reward contains only terms needed for the task:

- dense requested-face alignment
- cube-to-palm position retention
- angular settling only when the requested face is already within 30 degrees of world up
- small action penalty
- `+250` command-completion bonus
- `-50` drop penalty

There is no wrong-face penalty. The completion bonus is deliberately much larger than the dense reward so the policy cannot profit by remaining just below the hold threshold.

## Installation

Use the Python environment supplied by your Isaac Lab installation. RSL-RL should be installed through that Isaac Lab environment so its compatible version is used.

```bash
cd DICE
pip install -e .
```

For the optional OpenCV overlay:

```bash
pip install -e ".[video]"
```

## Train once

```bash
python scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --run_name final \
  --headless
```

The default RSL-RL setup is the official Shadow Hand PPO baseline:

- 16 steps per environment per update
- 10,000 maximum iterations
- checkpoint every 250 iterations
- actor and critic: `512 → 512 → 256 → 128`, ELU
- normalized actor and critic observations
- PPO clip `0.2`
- five learning epochs and four minibatches
- learning rate `5e-4` with adaptive KL schedule
- desired KL `0.016`
- `gamma=0.99`, `lambda=0.95`

Outputs are written under:

```text
outputs/DICE/<timestamp>_<run_name>/
```

The explicit final checkpoint is:

```text
model_final.pt
```

Resume toward the same total iteration target with:

```bash
python scripts/train_rsl.py \
  --resume outputs/DICE/<run>/model_5000.pt \
  --max_iterations 10000 \
  --run_name resumed \
  --headless
```

## Evaluate

Nominal evaluation:

```bash
python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Eval-v0 \
  --checkpoint outputs/DICE/<run>/model_final.pt \
  --episodes 500 \
  --num_envs 256 \
  --output evaluation/nominal \
  --headless
```

Held-out mass/friction evaluation:

```bash
python scripts/evaluate_rsl.py \
  --task DICE-Shadow-Robust-v0 \
  --checkpoint outputs/DICE/<run>/model_final.pt \
  --episodes 500 \
  --num_envs 256 \
  --output evaluation/robust \
  --headless
```

Run both with:

```bash
bash scripts/run_final_evaluation.sh outputs/DICE/<run>/model_final.pt
```

The evaluator writes:

- `episodes.csv`
- `summary.json`
- overall command success rate
- drop rate
- median command latency
- mean, median, and maximum commands per episode
- success rate for each requested face

## Render the final video

```bash
python scripts/play_rsl.py \
  --task DICE-Shadow-Play-v0 \
  --checkpoint outputs/DICE/<run>/model_final.pt \
  --output videos/DICE
```

The play environment uses the deterministic sequence:

```text
1 → 6 → 3 → 5 → 2 → 4
```

It uses the numbered die only for presentation. No mass or friction randomization is active during rendering.

Annotate the recorded MP4:

```bash
python scripts/annotate_video.py \
  --video videos/DICE/raw/<recorded-file>.mp4 \
  --metrics videos/DICE/video_metrics.csv \
  --output videos/DICE/DICE_annotated.mp4
```

## Recommended finish line

Do not launch new training variants before inspecting the checkpoints produced by the first 10,000-iteration run. Select a checkpoint using frozen nominal evaluation, then confirm it under the held-out robustness environment and render the deterministic six-command video.
