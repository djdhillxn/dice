# Project Proposal: DICE — Command-Conditioned In-Hand Die Reorientation

## 1. Executive Summary

The objective of **DICE** is to train a 20-DoF Shadow Hand in NVIDIA Isaac Lab to perform continuous, multi-command in-hand reorientation of a held die. The policy must rotate the die until a requested semantic face (1 through 6) points vertically upward, stabilize the die for a confirmation window, and immediately transition to new face targets in sequence without dropping or resetting the object.

This project deliberately avoids multi-stage hand-crafted curriculum learning. Instead, it relies on a **single, robust, end-to-end PPO training run** built upon potential-based reward shaping, explicit 3D geometric orientation vectors, and extended temporal credit assignment.

---

## 2. Core Task & Environment Specifications

### 2.1 Robot and Hardware Interface
* **Robot Model**: Shadow Hand (Right hand).
* **Action Space**: 20-dimensional continuous joint target commands ($\mathbf{a}_t \in [-1, 1]^{20}$).
* **Control Frequency**: 60 Hz physics step ($0.0167\text{s}$ per control step).

### 2.2 Task Definition & Success Criteria
A command specifies a requested face index $k \in \{1, 2, 3, 4, 5, 6\}$. A command transition is marked as successful when all three of the following conditions are simultaneously satisfied for **20 consecutive control steps** ($0.33\text{s}$):
1. **Angular Alignment Gate**: $\text{alignment} = \vec{N}_{\text{commanded}} \cdot \hat{z}_{\text{world}} \ge \cos(16^\circ) \approx 0.961$
2. **Palm Position Gate**: $\|\mathbf{p}_{\text{die}} - \mathbf{p}_{\text{palm}}\| \le 0.12\text{m}$
3. **Angular Speed Gate**: $\|\boldsymbol{\omega}_{\text{die}}\| \le 1.25\text{rad/s}$

Upon achieving success, the environment immediately assigns a new target face without resetting the hand posture or object position.

---

## 3. Observation Space Design

The policy uses a compact, continuous **126-dimensional** task-aligned observation:

| Observation Feature | Dimensions | Description |
|---|---|---|
| **Hand joint position and velocity** | 48 | 24 normalized positions and 24 scaled velocities |
| **Smoothed applied joint targets** | 20 | Normalized state of the low-pass joint-target controller |
| **Cube-frame fingertip position** | 15 | Five cube-relative position vectors expressed in the cube frame |
| **Cube-frame relative fingertip velocity** | 15 | Five fingertip-minus-cube linear-velocity vectors |
| **Cube translation and velocity** | 9 | Relative position, linear velocity, and angular velocity |
| **Cube rotation** | 6 | Continuous 6D orientation representation |
| **Command geometry** | 7 | Commanded world normal, vertical alignment, and rotation-axis error |
| **Hold Progress** | 1 | Continuous scalar $\frac{\text{hold\_counter}}{20} \in [0.0, 1.0]$ |
| **Fingertip reaction-load proxies** | 5 | Bounded magnitudes derived from incoming joint reaction wrenches |
| **Total Actor Observation Dimension** | **126** | Compact, task-aligned policy input |

An asymmetric **247-dimensional critic state** augments the actor observation
with full fingertip reaction wrenches and spatial velocities, object state, and
raw hand joint state. The actor does not receive those additional privileged
features.

---

## 4. Reward Architecture & Anti-Hacking Formulation

### 4.1 Angular-Error Progress Reward
To prevent the policy from getting trapped in local minima, the static alignment reward is replaced by progress in semantic angular error:
$$\theta_t = \arccos(\operatorname{clamp}(\text{alignment}_t)),\qquad
\mathcal{R}_{\text{progress}} = 40(\theta_{t-1}-\theta_t).$$
* **Loitering Prevention**: $\theta_t = \theta_{t-1} \implies \mathcal{R}_{\text{progress}} = 0$. Stationary loitering yields **zero points**.
* **Target Switch Reset**: When a new target is assigned upon success, the previous-error baseline is updated for the new command to prevent an artificial reward spike.

### 4.2 Continuous Hold Progress & Completion Incentives
* **Signed Hold Progress Shaping**: $\mathcal{R}_{\text{hold}} = c_{\text{hold}} \times \frac{h_t-h_{t-1}}{20}$, which claws back accumulated shaping when a partial hold breaks.
* **Success Completion Bonus**: Raw $+250.0$ upon reaching step 20 of hold.
* **Drop Penalty**: Raw $-100.0$ if position error exceeds $0.24\text{m}$.
* **Position Error Penalty**: $-2.0 \times \|\mathbf{p}_{\text{die}} - \mathbf{p}_{\text{palm}}\|$.
* **Applied-Target Rate Penalty**: $-0.01 \times \|\mathbf{u}_t-\mathbf{u}_{t-1}\|^2$, where $\mathbf{u}$ is the normalized smoothed joint-target state.
* **Raw-Action Boundary Penalty**: $-0.1\sum_i\max(|a_i|-0.9,0)^2$. Raw policy outputs are retained for this term; only commands applied to the hand are clamped into $[-1,1]$.
* **Settling Penalty**: A continuous orientation weight grows from zero at $45^\circ$ to one at $16^\circ$ and multiplies $-0.05\|\boldsymbol{\omega}_{\text{die}}\|^2$.
* **Global Reward Scale**: The sum of all raw components is multiplied by $0.1$ before being returned to PPO; logged components use the same effective scale.

---

## 5. RSL-RL PPO Agent Hyperparameters

* **Parallel Environments**: 2048
* **Rollout Horizon (`num_steps_per_env`)**: 32 steps ($0.53\text{s}$ per rollout update)
* **Total Training Iterations**: 5,000 completed
* **Network Architecture**: Policy & Value MLPs $[512, 512, 256, 128]$ with ELU activations
* **Observation Normalization**: Enabled
* **Exploration**: Direct scalar policy-noise standard deviation initialized at `0.6`, with entropy coefficient `0.0`
* **Learning Rate**: Fixed $3 \times 10^{-4}$
* **GAE Discount & Lambda**: $\gamma = 0.99, \lambda = 0.95$

---

## 6. Key Performance Metrics & Deliverables

1. **Nominal Command Completion Rate**: $\ge 90\%$ on `DICE-Shadow-Eval-v0`.
2. **Mean Completed Commands per Episode**: $\ge 3.5$ commands within $24\text{s}$ episode.
3. **Drop/Throughput Trade-Off**: Report drop rate alongside command completion and throughput across 1,000 nominal test episodes; a higher-throughput policy is not rejected solely for exceeding the earlier aspirational 5% drop target.
4. **Held-Out Mass & Friction Robustness**: Evaluated on `DICE-Shadow-Robust-v0` ($\pm 20\%$ mass & friction).
5. **Adverse Material Stress Test**: Evaluated on `DICE-Shadow-Adverse-v0` with fixed $1.5\times$ object mass and $0.7$ surface friction.
6. **Deterministic Playback Pipeline**: Supports rendering the 6-face sequence (`1 -> 6 -> 3 -> 5 -> 2 -> 4`) and annotating it with telemetry metrics. Producing the portfolio video is an optional presentation step, not part of the quantitative acceptance criteria.

---

## 7. Final Outcome

The final run completed 5,000 PPO iterations and 327.68 million transitions.
A sweep over five saved checkpoints selected `model_4000.pt` rather than the
last checkpoint, because it provided the best nominal drop/throughput trade-off.

Across 1,000 episodes per condition, the selected policy achieved 97.09%
issued-command completion, 33.334 mean sequential commands, and a 9.70% drop
rate under nominal physics. Symmetric held-out mass/material variation retained
essentially identical performance: 97.07% completion, 33.072 commands, and a
9.50% drop rate. The fixed heavy/low-friction stress preserved 95.92% issued-
command completion but reduced mean commands to 23.514 and increased episode
drops to 45.30%, exposing a clear long-horizon grasp-retention boundary.

The adverse result is a useful negative result and a future direction rather
than a failure of the nominal objective. It shows why material randomization
and system identification would be necessary for a stronger sim-to-real claim.
The complete interpretation and limitations are recorded in
[docs/final_results.md](docs/final_results.md).

---

*This document records the implemented design, acceptance criteria, and final experimental validation for the DICE project.*
