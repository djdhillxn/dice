# DICE — Direct Terminal Workflow & Architecture

**DICE** trains a Shadow Hand to reorient a held die in-hand to a requested numbered face and continue through new commands sequentially without releasing the object.

All workflows run **directly from the terminal using Python scripts**.

## Google Compute Engine / Conda setup

The GCE workflow uses the existing Conda environment named **`dice`** and should be launched from an SSH terminal in **headless** mode.

### Activate and verify the DICE environment

```bash
conda activate dice
cd ~/projects/dice
python --version
which python
```

The current repository does not contain an `environment.yml` / `environment.yaml`; the project-level Python dependencies are defined in `pyproject.toml`. Install or refresh the repository with:

```bash
python -m pip install -e .
```

For video annotation support:

```bash
python -m pip install -e ".[video]"
```

DICE pins `numpy==1.26.0` because Isaac Sim 5.1 requires that NumPy version, and pins `opencv-python-headless==4.11.0.86` because the VM does not need OpenCV GUI support. Do not install the latest OpenCV 5 wheel in this Isaac environment: on Python 3.11 it can require NumPy 2 and replace Isaac Sim's compatible NumPy.

If a previous editable install already upgraded NumPy to 2.x, repair the active `dice` environment once:

```bash
python -m pip uninstall -y \
  opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless

python -m pip install --upgrade \
  "numpy==1.26.0" \
  "opencv-python-headless==4.11.0.86"

python -m pip install -e ".[video]"
python -m pip check

python - <<'PY'
import cv2
import numpy
import sqlite3

print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("sqlite:", sqlite3.sqlite_version)
PY
```

The training script also refuses to start Isaac Sim if NumPy is not exactly `1.26.0`, so a future `pip install` cannot silently launch an unsupported runtime.

### If an environment YAML is added later

Create it under the same environment name:

```bash
conda env create -n dice -f environment.yml
conda activate dice
python -m pip install -e ".[video]"
```

If that YAML is changed later, update the existing environment rather than creating another one:

```bash
conda env update -n dice -f environment.yml --prune
conda activate dice
python -m pip install -e ".[video]"
```

Changes only to `pyproject.toml` do not require recreating the Conda environment; rerun the editable install.

### Fix `CXXABI_1.3.15` / `libstdc++.so.6` on Ubuntu 22.04

If Isaac Sim reports that `/lib/x86_64-linux-gnu/libstdc++.so.6` does not provide `CXXABI_1.3.15`, verify the Conda runtime:

```bash
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep CXXABI_1.3.15
```

If that symbol is missing:

```bash
conda install -y -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
```

Then make the active environment's libraries take precedence for this shell and verify `sqlite3` before launching Isaac Sim:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import sqlite3; print('sqlite OK:', sqlite3.sqlite_version)"
```

### Headless GCE training

```bash
python -u scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --run_name strong_run \
  --headless
```

`python -u` keeps terminal output unbuffered over SSH. The training script now prints and saves startup milestones before/after Gym creation, the RSL-RL wrapper reset, runner construction, and the PPO learning loop. If startup stalls, inspect the newest `outputs/<run>/startup.log` rather than guessing which layer is blocked.

The repeated `sh: 1: zenity: not found` message is not a request to install `zenity` for headless training. Always use `--headless` on the VM.

Before launching a paid full run after changing observations, actions, or the
critic state, run the two-iteration simulator contract preflight on the VM:

```bash
bash scripts/run_training_preflight.sh 64 2
```

The launcher verifies that the environment returns separate policy and critic
tensors with the configured dimensions before it constructs PPO storage. A
successful preflight must complete both PPO iterations and write a run with
`"status": "complete"` under `outputs/preflight/`.

### Training artifacts and TensorBoard

Every run writes to:

```text
outputs/<timestamp>_<run_name>/
```

The directory contains:

- `run.json`: command, complete environment/PPO configuration, Git state, hardware/software versions, timing, runtime status, and final checkpoint metadata.
- `startup.log`: persisted launch-stage breadcrumbs for startup debugging.
- `events.out.tfevents...`: TensorBoard scalars written by RSL-RL.
- `training_metrics.csv`: portable iteration-by-iteration scalar export.
- `training_summary.json`: first/last/extrema and last-100-iteration means for every scalar.
- `runtime_logs/`: Isaac Lab and Kit logs produced during this run, when available.
- `artifact_manifest.json`: complete artifact list and byte sizes for transfer auditing.
- `model_*.pt`: periodic RSL-RL checkpoints (`save_interval = 1000` by default; override with `--save_interval`).
- `model_final.pt`: explicit final/interrupted checkpoint from the DICE launcher.
- `git/*.diff`: repository state captured by RSL-RL on the first learning iteration, including DICE after the runner registers this repository.
- `evaluation/`: nominal and robust frozen-policy results produced by the evaluation scripts.

RSL-RL logs PPO losses, learning rate, policy noise, FPS/collection/learning time, reward, episode length, and every scalar under `extras["log"]`. DICE additionally records actor-mean action bounds, alignment, position and angular speed, all three success-gate rates, simultaneous-gate rate, hold-counter tails, command/drop statistics, every reward component, action magnitude/RMS, and action saturation.

`DICE/completion_frequency_per_env_step` is a frequency on the `[0, 1]` scale: it answers “what fraction of environments completed a command on this control step?” It is not a per-command success probability. Use frozen-policy evaluation for `target_face_success_rate`; that value is also stored on `[0, 1]` and should be multiplied by 100 for a percentage.

Start TensorBoard on the VM with:

```bash
tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

Forward port `6006` over SSH from your local machine and open `http://localhost:6006`. RSL-RL also prints its native detailed PPO summary after every learning iteration, so no custom chunked `runner.learn()` loop is required.

---

## 1. Terminal Workflow Overview

| Script | Purpose | CLI Command Example |
|---|---|---|
| `scripts/train_rsl.py` | Primary RSL-RL PPO training with live terminal progress bar & metrics | `python -u scripts/train_rsl.py --task DICE-Shadow-Train-v0 --num_envs 2048 --max_iterations 10000 --run_name strong_run --headless` |
| `scripts/run_training_preflight.sh` | Two-iteration actor/critic contract and PPO smoke test | `bash scripts/run_training_preflight.sh 64 2` |
| `scripts/evaluate_rsl.py` | Nominal or robust evaluation with progress bar & JSON/CSV outputs inside the checkpoint run | `python scripts/evaluate_rsl.py --task DICE-Shadow-Eval-v0 --checkpoint outputs/<run>/model_final.pt --episodes 500` |
| `scripts/run_final_evaluation.sh` | Runs both nominal and robust evaluations under the checkpoint run and generates `final_summary.json` | `bash scripts/run_final_evaluation.sh outputs/<run>/model_final.pt 500 256` |
| `scripts/play_rsl.py` | Renders continuous 6-face sequence (`1 -> 6 -> 3 -> 5 -> 2 -> 4`) | `python scripts/play_rsl.py --task DICE-Shadow-Play-v0 --checkpoint outputs/<run>/model_final.pt --output videos/DICE` |
| `scripts/annotate_video.py` | Overlays live telemetry metrics (target, top face, alignment, hold) onto rendered MP4 | `python scripts/annotate_video.py --video videos/DICE/raw/DICE.mp4 --metrics videos/DICE/video_metrics.csv --output videos/DICE/annotated.mp4` |

---

## 2. Theoretical Analysis & Reward Design

### Why Static Alignment Fails (The "Loitering" Problem)
In naive setups, policies receive a static posture reward proportional to how close the requested face is to pointing upward. For example, if a die is held at 45 degrees, the static alignment reward gives a constant positive signal every control step. Over a 24-second episode (1,440 steps), sitting stationary at 45 degrees yields **hundreds of reward points** for doing nothing! If the policy tries to flip the die further, it risks dropping it (-50 penalty). Consequently, the policy falls into a **local minimum of loitering indefinitely** without ever attempting to complete commands.

### Solutions Implemented

1. **Potential-Based Progress Reward (Delta Alignment)**:
   $$\text{Reward}_{\text{progress}} = c_{\text{progress}} \cdot (\text{alignment}_t - \text{alignment}_{t-1})$$
   - Staying stationary ($\text{alignment}_t = \text{alignment}_{t-1}$) yields **zero** progress reward. Loitering is completely unrewarded.
   - Rotating *toward* the target face yields positive progress reward.
   - Rotating *away* yields negative reward.

2. **Signed Hold Progress Shaping**:
   - Rewards each new valid hold step and claws the accumulated shaping back if the consecutive hold breaks:
     $$\text{Reward}_{\text{hold}} = c_{\text{hold}} \cdot \frac{h_t-h_{t-1}}{\text{hold\_steps}}$$
   - A partial hold cannot be repeatedly farmed for positive return.

3. **Command Completion Bonus (`+250`)**:
   - Because loitering earns zero points, completing commands to receive the `+250` success bonus is the **primary driver** of total episode return.

---

## 3. Environments

| Environment | Purpose | Object | Randomization |
|---|---|---|---|
| `DICE-Shadow-Train-v0` | Main PPO training | stock instanceable DexCube | none |
| `DICE-Shadow-Eval-v0` | Nominal evaluation | stock instanceable DexCube | none |
| `DICE-Shadow-Robust-v0` | Held-out robustness evaluation | stock instanceable DexCube | mass and friction ±20% |
| `DICE-Shadow-Play-v0` | Six-command video | local numbered die | none |

---

## 4. Actor and Critic Observations

The action space is Isaac Lab's 20-dimensional continuous Shadow Hand joint targets.

The policy receives a **126-dimensional task-aligned** observation space:

- **Hand proprioception** (48 dims): 24 normalized joint positions and 24 scaled joint velocities.
- **Applied controller state** (20 dims): Normalized smoothed joint targets currently applied to the hand.
- **Fingertip state** (30 dims): Five cube-frame fingertip positions and five cube-frame relative linear velocities.
- **Cube translation and velocity** (9 dims): Position relative to the nominal in-hand center plus linear and angular velocity.
- **Cube orientation** (6 dims): Continuous 6D rotation representation.
- **Command geometry** (7 dims): Commanded face normal in world coordinates, its alignment with world-up, and the cross-product rotation-axis error.
- **Hold progress** (1 dim): Normalized hold counter `hold_counter / 20`.
- **Fingertip load proxies** (5 dims): Bounded force magnitudes derived from incoming fingertip joint reaction wrenches.

The asymmetric critic receives a **247-dimensional privileged state** containing
the actor observation, full fingertip reaction wrenches and spatial velocities,
object pose and velocity, and raw hand joint state.

The 126-dimensional actor is not checkpoint-compatible with the earlier
121-dimensional policy. Start this configuration from a fresh policy. Evaluate
older checkpoints with the repository revision and observation contract that
created them.

---

## 5. RSL-RL PPO Configuration

- **Rollout Length**: `num_steps_per_env = 32` (65,536 transitions per update with 2,048 environments).
- **Optimizer**: Adam with a fixed `3e-4` learning rate.
- **Exploration**: Direct scalar standard deviation initialized at `0.6`, with no entropy bonus.
- **Networks**: Separate actor and critic MLPs with `[512, 512, 256, 128]`, ELU activations, and observation normalization.
- **Discount & GAE**: `gamma = 0.99`, `lambda = 0.95`.

---

## 6. Usage Examples

### Training
```bash
python scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --run_name strong_run \
  --headless
```

### Evaluation
```bash
bash scripts/run_final_evaluation.sh outputs/<timestamp>_strong_run/model_final.pt 500 256
```

### Rendering Video
```bash
python scripts/play_rsl.py \
  --task DICE-Shadow-Play-v0 \
  --checkpoint outputs/<timestamp>_strong_run/model_final.pt \
  --output videos/DICE
```

### Annotating Video
```bash
python scripts/annotate_video.py \
  --video videos/DICE/raw/DICE.mp4 \
  --metrics videos/DICE/video_metrics.csv \
  --output videos/DICE/DICE_annotated.mp4
```
