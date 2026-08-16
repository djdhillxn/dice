# DICE architecture

## Simulator layer

DICE subclasses Isaac Lab's direct in-hand manipulation environment. Isaac Lab retains responsibility for the Shadow Hand articulation, contact simulation, stock DexCube, vectorized resets, fingertip state, action smoothing, and joint-target application.

The custom environment adds only semantic face commands, yaw-invariant face alignment, hold-to-confirm completion, command switching, reward terms, and task diagnostics.

## Training object versus presentation object

Training, nominal evaluation, and robust evaluation use Isaac Lab's stock instanceable DexCube. This is the performance-sensitive and known-good object path.

The final adverse evaluation uses that same object and policy interface, but
fixes the object's mass at 1.5 times nominal and its static/dynamic friction at
0.7. This isolates a heavy, slippery material shift without changing geometry,
observations, commands, or success criteria.

The single-environment play configuration replaces it with the local numbered die. Its collision size and density are aligned with the stock cube configuration, but its visible pips are used only for the final video.

## Policy interface

The actor receives a 126-dimensional frame-consistent deployable observation:

```text
24 normalized hand joint positions
24 scaled hand joint velocities
20 normalized smoothed applied joint targets
15 relative fingertip positions in CUBE FRAME (5 x 3)
15 relative fingertip linear velocities in CUBE FRAME (5 x 3)
 3 relative cube position
 3 cube linear velocity
 3 cube angular velocity
 6 continuous 6D cube rotation
 3 commanded face normal (world frame)
 1 commanded face alignment (+Z)
 3 rotation axis error (cross product with +Z)
 1 normalized hold progress
 5 bounded fingertip reaction-load proxy magnitudes
---
126 total actor observation
```

The asymmetric critic receives a 247-dimensional privileged state observation:

```text
126 full actor observation
 30 fingertip 6D incoming joint reaction wrenches (body frame)
 30 fingertip 6D spatial velocities
  3 environment-local object position
  4 object rotation (world-frame wxyz quat)
  3 object linear velocity (world frame)
  3 object angular velocity (world frame)
 24 raw hand joint positions
 24 raw hand joint velocities
---
247 total critic state
```

The five actor load features are bounded magnitudes derived from the force
components of the fingertip incoming joint reaction wrenches. They are useful
grasp-load proxies, but they are not dedicated net-contact sensors.

The action space remains the 20-dimensional joint-target action. RSL-RL's
wrapper does not clip it: DICE retains the raw Gaussian policy output for the
boundary penalty and diagnostics, clamps the applied command into `[-1, 1]`,
and then passes that bounded command to the inherited smoothed controller.

## Command transition

A command is completed after 20 consecutive steps satisfying the final orientation, position, and angular-speed gates. The environment then selects a different face while preserving the current hand and cube state. This makes multi-command manipulation part of the training distribution rather than a separate curriculum stage.

## Runtime paths

```text
train_rsl.py
  DICE-Shadow-Train-v0
  RslRlVecEnvWrapper
  OnPolicyRunner
  .pt checkpoints

                    ┌─ evaluate_rsl.py → nominal metrics
.pt checkpoint ─────┼─ evaluate_rsl.py → symmetric robust metrics
                    ├─ evaluate_rsl.py → adverse material metrics
                    └─ play_rsl.py     → raw MP4 + overlay CSV
```
