# DICE — Direct Terminal Workflow & Architecture

**DICE** trains a Shadow Hand to reorient a held die in-hand to a requested numbered face and continue through new commands sequentially without releasing the object.

All workflows run **directly from the terminal using Python scripts**.

---

## 1. Terminal Workflow Overview

| Script | Purpose | CLI Command Example |
|---|---|---|
| `scripts/train_rsl.py` | Primary RSL-RL PPO training with live terminal progress bar & metrics | `python scripts/train_rsl.py --task DICE-Shadow-Train-v0 --num_envs 2048 --max_iterations 10000 --run_name strong_run` |
| `scripts/evaluate_rsl.py` | Nominal or robust evaluation with progress bar & JSON/CSV outputs | `python scripts/evaluate_rsl.py --task DICE-Shadow-Eval-v0 --checkpoint outputs/DICE/<run>/model_final.pt --episodes 500` |
| `scripts/run_final_evaluation.sh` | Runs both nominal and robust evaluations and generates `final_summary.json` | `bash scripts/run_final_evaluation.sh outputs/DICE/<run>/model_final.pt 500 256` |
| `scripts/play_rsl.py` | Renders continuous 6-face sequence (`1 -> 6 -> 3 -> 5 -> 2 -> 4`) | `python scripts/play_rsl.py --task DICE-Shadow-Play-v0 --checkpoint outputs/DICE/<run>/model_final.pt --output videos/DICE` |
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

2. **Continuous Hold Progress Shaping**:
   - Rewards climbing toward the 20-step hold gate smoothly while inside the 16-degree success zone:
     $$\text{Reward}_{\text{hold}} = c_{\text{hold}} \cdot \frac{\text{hold\_counter}}{\text{hold\_steps}}$$

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

## 4. Observation Space (174 Dimensions)

The action space is Isaac Lab's 20-dimensional continuous Shadow Hand joint targets.

The policy receives a **174-dimensional** observation space:

- **Stock Shadow Hand full observation** (157 dims): Joint positions, velocities, hand pose, object pose, object velocities.
- **Requested face one-hot** (6 dims): One-hot encoding of target face 1..6.
- **Hold progress** (1 dim): Normalized hold counter `hold_counter / 20`.
- **Target face alignment** (1 dim): Scalar dot product of commanded face normal with world UP `(0,0,1)`.
- **Commanded face normal in world frame** (3 dims): 3D unit vector $[N_x, N_y, N_z]$.
- **Current top face normal in world frame** (3 dims): 3D unit vector $[n_x, n_y, n_z]$.
- **Rotation axis error vector** (3 dims): Cross product $\vec{n} \times \vec{N}$ (instantaneous 3D rotation axis required to align the faces).

---

## 5. RSL-RL PPO Configuration

- **Rollout Length**: `num_steps_per_env = 64` (provides a long credit assignment horizon across simulation steps).
- **Optimizer**: Adam with adaptive KL learning rate schedule (`desired_kl = 0.016`, initial LR `5e-4`).
- **Network**: Shared depth `[512, 512, 256, 128]` with ELU activations and observation normalization.
- **Discount & GAE**: `gamma = 0.99`, `lambda = 0.95`.

---

## 6. Usage Examples

### Training
```bash
python scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 10000 \
  --run_name strong_run
```

### Evaluation
```bash
bash scripts/run_final_evaluation.sh outputs/DICE/<timestamp>_strong_run/model_final.pt 500 256
```

### Rendering Video
```bash
python scripts/play_rsl.py \
  --task DICE-Shadow-Play-v0 \
  --checkpoint outputs/DICE/<timestamp>_strong_run/model_final.pt \
  --output videos/DICE
```

### Annotating Video
```bash
python scripts/annotate_video.py \
  --video videos/DICE/raw/DICE.mp4 \
  --metrics videos/DICE/video_metrics.csv \
  --output videos/DICE/DICE_annotated.mp4
```
