# DICE Dial — Continuous Dexterous In-Hand Die Reorientation

**DICE Dial** trains a 20-DoF Shadow Hand in NVIDIA Isaac Lab to rotate a held die to requested numbered faces and continue through new commands **without resetting the hand or object between successes**.

The final policy was trained with PPO over **327.68 million transitions**, selected by a nominal checkpoint sweep before the final test panels, and evaluated for **1,000 episodes per condition** under nominal physics, unseen ±20% object mass/friction variation, and a deliberately adverse heavy/low-friction corner.

## Final results

| Metric | Nominal | Held-out ±20% physics | Adverse: 1.5× mass, 0.7× friction |
|---|---:|---:|---:|
| Issued-command completion | **97.09%** | **97.07%** | **95.92%** |
| Episode drop rate | **9.70%** | **9.50%** | **45.30%** |
| Mean completed commands / episode | 33.334 | 33.072 | 23.514 |
| Median completed commands / episode | 37 | 37 | 32 |
| Commands / simulated minute | 90.536 | 89.000 | 81.996 |
| Median command latency | 0.617 s | 0.617 s | 0.650 s |
| Minimum per-face completion | 96.88% | 96.80% | 95.11% |
| Deterministic actor-mean OOB rate | 20.77% | 20.77% | 20.91% |

The moderate robustness condition is a genuine **held-out dynamics test**: training used nominal object physics with no mass/friction randomization. Performance under the ±20% distribution was statistically indistinguishable from nominal on the reported drop-rate and throughput comparisons. The fixed adverse corner preserved rapid per-command reorientation but exposed a long-horizon grasp-retention boundary: 424 of 453 dropped episodes completed at least one command first, and 219 completed at least ten.

`issued-command completion` is not a one-shot probability of never dropping the die. Every completed command counts as a successful attempt, and the command active when an episode terminates counts as one unfinished attempt:

\[
\text{issued-command completion}
=
\frac{\text{completed commands}}
{\text{completed commands}+\text{episodes}}.
\]

Read it together with drop rate, completed commands per episode, and latency. The full statistical interpretation, failure decomposition, and limitations are in [`docs/final_results.md`](docs/final_results.md).

---

## Task formulation

A command requests one die face \(k\in\{1,\dots,6\}\) to point upward. If the current die rotation is \(R\in SO(3)\) and \(\mathbf n_k\) is the requested face normal in object coordinates, then

\[
\mathbf n_{\text{world}}=R\mathbf n_k,
\qquad
\text{alignment}=\mathbf n_{\text{world}}\cdot \hat{\mathbf z}.
\]

A command completes only after **20 consecutive 60 Hz control steps** satisfying all three gates:

- orientation error \(\le 16^\circ\), equivalently alignment \(\ge \cos(16^\circ)\approx 0.961\);
- die position error \(\le 0.12\text{ m}\);
- die angular speed \(\le 1.25\text{ rad/s}\).

The environment then immediately issues a different face command while preserving the current hand and object state. This continuous command switching is part of the task from the first training transition; there is no staged task progression.

## Reward design: avoiding loitering

A static alignment reward can produce a simple failure mode: a policy reaches a partially aligned pose and stays there because the same positive posture reward is collected every step while attempting the final rotation risks a drop.

DICE Dial instead rewards **change in angular error**:

\[
\theta_t=\arccos(\operatorname{clamp}(\text{alignment}_t)),
\qquad
r_{\text{progress}}=40(\theta_{t-1}-\theta_t).
\]

Standing still therefore produces exactly zero progress reward. The remaining shaping terms are deliberately tied to task progress rather than static occupancy:

- **Signed hold progress:** \(40(h_t-h_{t-1})/20\), so breaking a partial confirmation hold claws back the accumulated shaping.
- **Command completion:** raw `+250` bonus.
- **Drop:** raw `-100` penalty when the object leaves the allowed in-hand region.
- **Position / settling penalties:** discourage drift and excessive angular speed near the target.
- **Applied-target rate penalty:** discourages unnecessarily abrupt controller changes.
- **Raw-action boundary penalty:** penalizes Gaussian policy outputs beyond `|a| > 0.9`; the command actually applied to the hand is clamped to `[-1, 1]`.
- **Global reward scale:** `0.1` is applied to the summed raw reward before PPO.

This is **differential progress shaping**. The project does not rely on a claim that the full shaped reward is an exact policy-invariant potential transform.

---

## Policy architecture

DICE Dial uses an **asymmetric actor-critic**: the actor receives a compact task-facing observation, while the critic may use additional simulator state during training.

### Actor: 126 dimensions

| Observation group | Dims | Contents |
|---|---:|---|
| Hand proprioception | 48 | 24 normalized joint positions + 24 scaled joint velocities |
| Applied controller state | 20 | Smoothed joint targets currently applied to the hand |
| Cube-frame fingertip kinematics | 30 | Five relative positions + five relative linear velocities |
| Cube translation / velocity | 9 | Relative position, linear velocity, angular velocity |
| Cube orientation | 6 | Continuous 6D rotation representation |
| Command geometry | 7 | Requested world normal, alignment, rotation-axis error |
| Hold progress | 1 | `hold_counter / 20` |
| Fingertip load proxies | 5 | Bounded magnitudes derived from fingertip reaction wrenches |
| **Total** | **126** | |

### Critic: 247 dimensions

The critic receives the full actor observation plus privileged fingertip 6D reaction wrenches, fingertip spatial velocities, object state, and raw hand joint state. Actor and critic are separate `[512, 512, 256, 128]` MLPs with **ELU activations and input normalization**.

The actor excludes the critic's extra privileged state, but this should not be read as a completed real-robot deployment interface: real hardware would still need reliable object-state estimation and compatible fingertip/load sensing.

The action space is the inherited **20-dimensional Shadow Hand joint-target command**.

See [`docs/architecture.md`](docs/architecture.md) for the exact observation contract and simulation/presentation object distinction.

---

## Training

| Item | Final configuration |
|---|---|
| Simulator | NVIDIA Isaac Lab / Isaac Sim |
| RL library | RSL-RL PPO |
| Training environments | 2,048 |
| Control frequency | 60 Hz (`dt = 1/120 s`, decimation 2) |
| Rollout length | 32 steps / environment |
| PPO iterations | 5,000 |
| Total transitions | 327,680,000 |
| Actor / critic MLP | `[512, 512, 256, 128]`, ELU |
| Initial policy noise | 0.6 learned scalar std |
| Learning rate | `3e-4`, fixed |
| Discount / GAE | `gamma = 0.99`, `lambda = 0.95` |
| PPO clip | 0.2 |
| Entropy coefficient | 0.0 |
| Training seed | 42 |
| Hardware | NVIDIA L4 |
| Training physics randomization | **None** |

The final run used Isaac Lab `2.3.2.post1`, RSL-RL `3.1.2`, PyTorch `2.7.0+cu128`, Python `3.11.15`, and CUDA `12.8`.

### Checkpoint selection

Five saved policies were screened on **500 nominal episodes each** before the final three-condition evaluation:

| Checkpoint | Completed commands / ep | Issued completion | Drop rate | Median latency | Status |
|---|---:|---:|---:|---:|---|
| `model_1000.pt` | 17.62 | 94.63% | 22.40% | 0.950 s | Candidate |
| `model_2000.pt` | 27.84 | 96.53% | 14.80% | 0.700 s | Candidate |
| `model_3000.pt` | 31.98 | 96.97% | 11.20% | 0.633 s | Candidate |
| **`model_4000.pt`** | **33.42** | **97.10%** | **9.40%** | **0.617 s** | **Selected** |
| `model_final.pt` | 32.89 | 96.93% | 14.00% | 0.583 s | Higher drop rate |

`model_4000.pt` was selected **before final testing** because it gave the best nominal retention/throughput trade-off. Training longer under the same recipe did not monotonically improve safety.

---

## Final evaluation

The frozen `model_4000.pt` checkpoint was evaluated for **1,000 full 24-second episodes per condition** using 256 concurrent environments:

1. **Nominal** — stock DexCube, nominal material parameters.
2. **Held-out ±20% physics** — mass, static friction, and dynamic friction sampled within `[0.8, 1.2]` of nominal; dynamic friction is constrained not to exceed static friction.
3. **Adverse stress** — fixed `1.5×` object mass and `0.7×` object static/dynamic friction.

The adverse test changes mass and friction together, so it identifies a useful stress boundary but **not** the separate causal contribution of each parameter. Under this condition:

| Adverse drop decomposition | Result |
|---|---:|
| Dropped episodes | 453 / 1,000 |
| Drops after at least 1 completed command | 424 |
| Drops after at least 10 completed commands | 219 |
| Drops after at least 20 completed commands | 109 |
| Median completed commands before drop | 9 |
| Median time before drop | 7.28 s |
| Mean commands in episodes surviving to timeout | 33.47 |

The result is best described as **simulation robustness to held-out object-material variation with a measured adverse boundary**. It is relevant to sim-to-real methodology, but no real-hardware transfer is claimed.

---

## Portfolio videos

The final presentation package contains three replay-validated 1920×1080 H.264 videos at **0.5× playback**:

| Story | What it shows |
|---|---|
| `dice_nominal_success.mp4` | 12-command nominal rollout, synchronized oblique + top views |
| `dice_physics_variation.mp4` | Same policy / face sequence under nominal vs held-out ±20% physics |
| `dice_adverse_boundary.mp4` | Representative heavy/low-friction rollout through its eventual drop |

Representative seed `9` is fixed across the presentation package. These videos are illustrative; all quantitative claims above come from the aggregate 1,000-episode evaluations.

The renderer verifies checkpoint hashes, audits the numbered presentation die against the stock evaluation object's mass/inertia, replays identical action trajectories across camera views, checks telemetry synchronization, creates event-selected WebP posters, and writes checksums plus a machine-readable manifest.

Run the complete renderer with:

```bash
python -u scripts/render_portfolio_videos.py \
  <timestamp>_<run_name> \
  --force
```

See [`docs/video_rendering.md`](docs/video_rendering.md) for the full evidence and rendering contract.

---

## Reproducing the workflow

### Install the project

The Isaac Lab / Isaac Sim runtime must already be available. Inside the DICE Conda environment:

```bash
python -m pip install -e .
```

For portfolio rendering:

```bash
python -m pip install -e ".[video]"
```

The project pins `numpy==1.26.0` and `opencv-python-headless==4.11.0.86` for the final Isaac Sim 5.1 environment. VM setup, compatibility notes, the `CXXABI_1.3.15` repair, TensorBoard, and GCE-specific commands are intentionally kept out of this README; see [`docs/gcp_setup.md`](docs/gcp_setup.md).

### Train

```bash
python -u scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 5000 \
  --run_name angular_bound_pilot_gurgaon \
  --headless
```

Before a paid full run after changing the actor, critic, observation, action, or controller contract:

```bash
bash scripts/run_training_preflight.sh 64 2
```

### Evaluate

```bash
python -u scripts/run_checkpoint_sweep.py <timestamp>_<run_name>
```

Then run the three final conditions for the selected checkpoint:

```bash
bash scripts/run_final_evaluation.sh \
  outputs/<run>/model_4000.pt \
  1000 \
  256
```

### Key scripts

| Script | Purpose |
|---|---|
| `scripts/train_rsl.py` | RSL-RL PPO training + provenance |
| `scripts/run_training_preflight.sh` | Actor/critic simulator contract check |
| `scripts/run_checkpoint_sweep.py` | Nominal screening and checkpoint ranking |
| `scripts/evaluate_rsl.py` | Single-condition frozen-policy evaluation |
| `scripts/run_final_evaluation.sh` | Reproducible three-condition final evaluation |
| `scripts/play_rsl.py` | Deterministic rollout capture and trajectory replay |
| `scripts/render_portfolio_videos.py` | End-to-end final presentation pipeline |
| `scripts/annotate_video.py` | Synchronized HUD and H.264 composition |

---

## Repository guide

| Document | Purpose |
|---|---|
| [`project_proposal.md`](project_proposal.md) | Implemented task design and final objective |
| [`docs/architecture.md`](docs/architecture.md) | Exact actor/critic, simulator, and object contracts |
| [`docs/experiment_plan.md`](docs/experiment_plan.md) | Preflight, checkpoint selection, and evaluation protocol |
| [`docs/final_results.md`](docs/final_results.md) | Final statistics, uncertainty, limitations, and closure decision |
| [`docs/video_rendering.md`](docs/video_rendering.md) | Reproducible video evidence pipeline |
| [`docs/gcp_setup.md`](docs/gcp_setup.md) | GCE / Conda / Isaac runtime notes |
| [`docs/references.md`](docs/references.md) | Papers, software, and physics references |
| [`docs/final_comparison.csv`](docs/final_comparison.csv) | Compact tracked final evaluation table |

Large checkpoints, per-episode evaluation records, raw captures, and generated videos live under ignored `outputs/` and `videos/` directories; the tracked documentation records their exact provenance and summaries.

---

## Limitations and next steps

- The policy was trained with **one PPO seed**. Evaluation confidence intervals therefore measure episode-sampling uncertainty, not training-seed variability.
- The final conditions use fixed evaluation seeds and are reproducible, but they are not paired trajectory-by-trajectory.
- Robustness testing covers object mass and friction only. Observation noise, latency, actuator mismatch, contact compliance, geometry variation, and real hardware were not tested.
- The adverse condition changes mass and friction jointly and should not be interpreted as a mass-vs-friction ablation.
- The deterministic Gaussian actor mean is outside `[-1,1]` for about **20.8%** of action dimensions. Applied commands are clamped before reaching the hand, and the rate is nearly unchanged across all three evaluation conditions; it remains an implementation diagnostic worth addressing in a future policy iteration.

A substantive extension would train the same full task with physically plausible dynamics randomization, then reevaluate the unchanged adverse corner, add separate mass-only/friction-only stress tests, repeat training across multiple seeds, and finally perform real-object system identification before making a sim-to-real claim.

---

## References and foundations

DICE Dial builds directly on the following work and software:

1. **Isaac Lab** — Mittal et al., *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning* ([paper](https://arxiv.org/abs/2511.04831), [project](https://github.com/isaac-sim/IsaacLab)).
2. **RSL-RL** — Schwarke et al., *RSL-RL: A Learning Library for Robotics Research* ([paper](https://arxiv.org/abs/2509.10771), [project](https://github.com/leggedrobotics/rsl_rl)).
3. **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms* ([paper](https://arxiv.org/abs/1707.06347)).
4. **Asymmetric actor-critic** — Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning* ([paper](https://arxiv.org/abs/1710.06542)).
5. **Dexterous Shadow Hand RL** — Andrychowicz et al., *Learning Dexterous In-Hand Manipulation* ([paper](https://arxiv.org/abs/1808.00177)).
6. **Sequential dexterous manipulation / sim-to-real** — Akkaya et al., *Solving Rubik's Cube with a Robot Hand* ([paper](https://arxiv.org/abs/1910.07113)).
7. **Continuous rotation representations** — Zhou et al., *On the Continuity of Rotation Representations in Neural Networks* ([paper](https://arxiv.org/abs/1812.07035)).
8. **PhysX contact dynamics** — NVIDIA PhysX documentation on [rigid-body friction](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/RigidBodyDynamics.html) and [material combine modes](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html).

A larger, annotated reading list covering dexterous manipulation, dynamics randomization, reward shaping, GAE, Isaac Sim, and related foundations is in [`docs/references.md`](docs/references.md).

## License

MIT — see [`LICENSE`](LICENSE).
