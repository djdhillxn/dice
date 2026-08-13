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

## 3. Observation Space Design (121 Dimensions)

The policy uses a compact, continuous **121-dimensional** task-aligned observation:

| Observation Feature | Dimensions | Description |
|---|---|---|
| **Hand joint position and velocity** | 48 | 24 normalized positions and 24 scaled velocities |
| **Previous action** | 20 | Joint command applied on the preceding control step |
| **Cube-relative fingertip position** | 15 | Five fingertip position vectors |
| **Fingertip linear velocity** | 15 | Five world-frame linear-velocity vectors |
| **Cube translation and velocity** | 9 | Relative position, linear velocity, and angular velocity |
| **Cube rotation** | 6 | Continuous 6D orientation representation |
| **Command geometry** | 7 | Commanded world normal, vertical alignment, and rotation-axis error |
| **Hold Progress** | 1 | Continuous scalar $\frac{\text{hold\_counter}}{20} \in [0.0, 1.0]$ |
| **Total Observation Dimension** | **121** | Compact, task-aligned policy input |

---

## 4. Reward Architecture & Anti-Hacking Formulation

### 4.1 Potential-Based Progress Reward (Delta Alignment)
To prevent the policy from getting trapped in local minima (loitering stationary at partial angles like $45^\circ$ for positive points), the static alignment reward is replaced with **Potential-Based Progress Shaping**:
$$\mathcal{R}_{\text{progress}} = c_{\text{progress}} \times (\text{alignment}_t - \text{alignment}_{t-1})$$
* **Loitering Penalty**: $\text{alignment}_t = \text{alignment}_{t-1} \implies \mathcal{R}_{\text{progress}} = 0$. Stationary loitering yields **zero points**.
* **Target Switch Reset**: When a new target is assigned upon success, $\text{alignment}_{t-1}$ is updated to the new target's baseline alignment to prevent artificial negative delta spikes.

### 4.2 Continuous Hold Progress & Completion Incentives
* **Signed Hold Progress Shaping**: $\mathcal{R}_{\text{hold}} = c_{\text{hold}} \times \frac{h_t-h_{t-1}}{20}$, which claws back accumulated shaping when a partial hold breaks.
* **Success Completion Bonus**: $+250.0$ upon reaching step 20 of hold. Primary driver of return.
* **Drop Penalty**: $-50.0$ if position error exceeds $0.24\text{m}$.
* **Position Error Penalty**: $-2.0 \times \|\mathbf{p}_{\text{die}} - \mathbf{p}_{\text{palm}}\|$.
* **Action-Rate Penalty**: $-0.01 \times \|\mathbf{a}_t-\mathbf{a}_{t-1}\|^2$; there is no separate action-magnitude penalty.

---

## 5. RSL-RL PPO Agent Hyperparameters

* **Parallel Environments**: 2048
* **Rollout Horizon (`num_steps_per_env`)**: 32 steps ($0.53\text{s}$ per rollout update)
* **Total Training Iterations**: 10,000
* **Network Architecture**: Policy & Value MLPs $[512, 512, 256, 128]$ with ELU activations
* **Observation Normalization**: Enabled
* **Exploration**: Initial policy noise standard deviation `0.8` and entropy coefficient `0.005`
* **Learning Rate**: $5 \times 10^{-4}$ with adaptive KL schedule (`desired_kl = 0.016`)
* **GAE Discount & Lambda**: $\gamma = 0.99, \lambda = 0.95$

---

## 6. Key Performance Metrics & Deliverables

1. **Nominal Command Completion Rate**: $\ge 90\%$ on `DICE-Shadow-Eval-v0`.
2. **Mean Completed Commands per Episode**: $\ge 3.5$ commands within $24\text{s}$ episode.
3. **Drop Rate**: $\le 5\%$ across 500 test episodes.
4. **Held-Out Mass & Friction Robustness**: Evaluated on `DICE-Shadow-Robust-v0` ($\pm 20\%$ mass & friction).
5. **Deterministic Playback & Video Artifacts**: Rendered 6-face sequence (`1 -> 6 -> 3 -> 5 -> 2 -> 4`) annotated with telemetry metrics.

---

*This document serves as the immutable anchor for all codebase implementations and experimental validations in the DICE project.*
