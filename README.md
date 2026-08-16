# DICE — Direct Terminal Workflow & Architecture

**DICE** trains a Shadow Hand to reorient a held die in-hand to a requested numbered face and continue through new commands sequentially without releasing the object.

All workflows run **directly from the terminal using Python scripts**.

## Final outcome

The completed experiment trained for 5,000 PPO iterations (327.68 million
transitions) and selected `model_4000.pt` by a five-checkpoint nominal sweep.
The frozen policy was then evaluated for 1,000 episodes in each final
condition:

| Metric | Nominal | Symmetric physics variation | Heavy / low-friction stress |
|---|---:|---:|---:|
| Issued-command completion | 97.09% | 97.07% | 95.92% |
| Episode drop rate | 9.70% | 9.50% | 45.30% |
| Mean completed commands / episode | 33.334 | 33.072 | 23.514 |
| Median completed commands / episode | 37 | 37 | 32 |
| Commands / simulated minute | 90.536 | 89.000 | 81.996 |
| Median command latency | 0.617 s | 0.617 s | 0.650 s |
| Minimum per-face completion | 96.88% | 96.80% | 95.11% |
| Deterministic action OOB rate | 20.77% | 20.77% | 20.91% |

The moderate held-out physics distribution—object mass and material
coefficients sampled within `[0.8, 1.2]` of nominal, with dynamic friction
constrained not to exceed static friction—produced no statistically resolvable
loss relative to nominal evaluation. The fixed adverse corner (`1.5x` mass and
`0.7` object friction) preserved fast command completion but increased drops
to 45.3%. This is a long-horizon retention failure: 424 of the 453 adverse
drop episodes completed at least one command before dropping, and 219
completed at least ten.

`issued-command completion` is not a one-shot success probability. Every
completed command counts as a successful attempt and the command active at
episode termination counts as one unfinished attempt. It must therefore be
read together with drop rate, command throughput, and latency. See
[the full final report](docs/final_results.md) for provenance, uncertainty,
failure decomposition, limitations, and the project-closure decision.

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
  --max_iterations 5000 \
  --run_name angular_bound_pilot_gurgaon \
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
- `evaluation/`: checkpoint selection plus nominal, symmetric-robust, and adverse frozen-policy results.

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
| `scripts/train_rsl.py` | Primary RSL-RL PPO training with live terminal progress bar & metrics | `python -u scripts/train_rsl.py --task DICE-Shadow-Train-v0 --num_envs 2048 --max_iterations 5000 --run_name angular_bound_pilot_gurgaon --headless` |
| `scripts/run_training_preflight.sh` | Two-iteration actor/critic contract and PPO smoke test | `bash scripts/run_training_preflight.sh 64 2` |
| `scripts/evaluate_rsl.py` | Single-condition nominal, robust, or adverse evaluation with progress and JSON/CSV outputs | `python scripts/evaluate_rsl.py --task DICE-Shadow-Eval-v0 --checkpoint outputs/<run>/model_4000.pt --episodes 1000` |
| `scripts/run_checkpoint_sweep.py` | Discovers, nominally evaluates, and ranks every saved checkpoint except `model_0.pt` | `python -u scripts/run_checkpoint_sweep.py <timestamp>_<run_name>` |
| `scripts/run_final_evaluation.sh` | Runs the three final conditions, supports interruption-safe reuse, and writes combined JSON/CSV/text results | `bash scripts/run_final_evaluation.sh outputs/<run>/model_4000.pt` |
| `scripts/play_rsl.py` | Renders continuous 6-face sequence (`1 -> 6 -> 3 -> 5 -> 2 -> 4`) | `python scripts/play_rsl.py --task DICE-Shadow-Play-v0 --checkpoint outputs/<run>/model_4000.pt --output videos/DICE` |
| `scripts/annotate_video.py` | Overlays live telemetry metrics (target, top face, alignment, hold) onto rendered MP4 | `python scripts/annotate_video.py --video videos/DICE/raw/DICE.mp4 --metrics videos/DICE/video_metrics.csv --output videos/DICE/annotated.mp4` |

---

## 2. Theoretical Analysis & Reward Design

### Why Static Alignment Fails (The "Loitering" Problem)
In naive setups, policies receive a static posture reward proportional to how close the requested face is to pointing upward. For example, if a die is held at 45 degrees, the static alignment reward gives a constant positive signal every control step. Over a 24-second episode (1,440 steps), sitting stationary at 45 degrees yields **hundreds of reward points** for doing nothing. If the policy tries to flip the die further, it risks dropping it. Consequently, the policy can fall into a **local minimum of loitering indefinitely** without ever attempting to complete commands.

### Solutions Implemented

1. **Angular-Error Progress Reward**:
   $$\theta_t = \arccos(\operatorname{clamp}(\text{alignment}_t)),\qquad
   \text{Reward}_{\text{progress}} = 40(\theta_{t-1}-\theta_t)$$
   - Staying stationary yields **zero** progress reward.
   - Angular progress has a consistent scale across difficult and near-target orientations.
   - Rotating *toward* the target face yields positive reward; rotating *away* yields negative reward.

2. **Signed Hold Progress Shaping**:
   - Rewards each new valid hold step and claws the accumulated shaping back if the consecutive hold breaks:
     $$\text{Reward}_{\text{hold}} = c_{\text{hold}} \cdot \frac{h_t-h_{t-1}}{\text{hold\_steps}}$$
   - A partial hold cannot be repeatedly farmed for positive return.

3. **Command Completion and Drop Terms**:
   - Command completion earns a raw `+250`; dropping incurs a raw `-100`.
   - A global reward scale of `0.1` makes their effective PPO rewards `+25` and `-10` while preserving the intended relative incentives.

4. **Raw-Action Boundary Penalty**:
   - The environment stores the unbounded Gaussian policy output, clamps only the command applied to the hand, and penalizes squared excess beyond `|a| = 0.9`.
   - This gives PPO a learning signal against action-clipping aliasing while preserving bounded physical joint targets.

---

## 3. Environments

| Environment | Purpose | Object | Randomization |
|---|---|---|---|
| `DICE-Shadow-Train-v0` | Main PPO training | stock instanceable DexCube | none |
| `DICE-Shadow-Eval-v0` | Nominal evaluation | stock instanceable DexCube | none |
| `DICE-Shadow-Robust-v0` | Symmetric held-out robustness evaluation | stock instanceable DexCube | mass and physically consistent friction ±20% |
| `DICE-Shadow-Adverse-v0` | Adverse material stress evaluation | stock instanceable DexCube | fixed 1.5x mass and 0.7 friction |
| `DICE-Shadow-Play-v0` | Six-command video | local numbered die | none |

---

## 4. Actor and Critic Observations

The action space is a 20-dimensional continuous Shadow Hand joint target command. The Gaussian policy output is retained for PPO and boundary-penalty accounting, while the environment clamps the command applied to the hand into `[-1, 1]`.

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
- **Action application**: RSL-RL wrapper clipping is disabled; `DiceEnv` records raw actions and clamps only the applied controller command.
- **Networks**: Separate actor and critic MLPs with `[512, 512, 256, 128]`, ELU activations, and observation normalization.
- **Discount & GAE**: `gamma = 0.99`, `lambda = 0.95`.

---

## 6. Usage Examples

### Training
```bash
python scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 5000 \
  --run_name angular_bound_pilot_gurgaon \
  --headless
```

### Evaluation
```bash
bash scripts/run_final_evaluation.sh \
  outputs/<timestamp>_strong_run/model_4000.pt \
  1000 \
  256
```

The shell wrapper evaluates nominal physics with seed `2026`, symmetric held-out
mass/friction with seed `2027`, and the fixed heavy/slippery stress condition
with seed `2028`. Its stable default output is
`evaluation/final_<checkpoint-stem>/`; matching completed 1,000-episode
conditions are reused after an interruption. Pass `--force` as the fifth
argument (or in place of the output-directory argument) to rerun every
condition. Existing 500-episode summaries fail the reuse contract and are
overwritten in place, so no manual cleanup or extra timestamp directory is
needed. The directory contains per-condition CSV and JSON files plus
`evaluation_run.json`, `final_summary.json`, `final_comparison.csv`, and
`final_comparison.txt`.

### Checkpoint Sweep

Evaluate every periodic checkpoint plus `model_final.pt` sequentially, keep each
checkpoint's artifacts separate, and rank them by the documented acceptance
criteria:

```bash
python -u scripts/run_checkpoint_sweep.py <timestamp>_<run_name>
```

The command accepts either the run ID below `outputs/` or a direct run-directory
path. Completed matching evaluations are reused after an interruption; pass
`--force` to rerun them. Results are written to
`outputs/<run>/evaluation/checkpoint_sweep/`, including `ranking.json`,
`ranking.csv`, `ranking.txt`, and `selected_checkpoint.txt`.

### Rendering Video
```bash
python scripts/play_rsl.py \
  --task DICE-Shadow-Play-v0 \
  --checkpoint outputs/<timestamp>_strong_run/model_4000.pt \
  --output videos/DICE
```

### Annotating Video
```bash
python scripts/annotate_video.py \
  --video videos/DICE/raw/DICE.mp4 \
  --metrics videos/DICE/video_metrics.csv \
  --output videos/DICE/DICE_annotated.mp4
```
