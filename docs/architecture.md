# DICE architecture

## Simulator layer

DICE subclasses Isaac Lab's direct in-hand manipulation environment. Isaac Lab retains responsibility for the Shadow Hand articulation, contact simulation, stock DexCube, vectorized resets, fingertip state, action smoothing, and joint-target application.

The custom environment adds only semantic face commands, yaw-invariant face alignment, hold-to-confirm completion, command switching, reward terms, and task diagnostics.

## Training object versus presentation object

Training, nominal evaluation, and robust evaluation use Isaac Lab's stock instanceable DexCube. This is the performance-sensitive and known-good object path.

The single-environment play configuration replaces it with the local numbered die. Its collision size and density are aligned with the stock cube configuration, but its visible pips are used only for the final video.

## Policy interface

The actor receives a clean 121-dimensional task-aligned observation:

```text
24 normalized hand joint positions
24 scaled hand joint velocities
20 previous actions
15 relative fingertip positions (5 x 3)
15 fingertip linear velocities (5 x 3)
 3 relative cube position
 3 cube linear velocity
 3 cube angular velocity
 6 continuous 6D cube rotation
 3 commanded face normal (world frame)
 1 commanded face alignment (+Z)
 3 rotation axis error (cross product with +Z)
 1 normalized hold progress
---
121 total
```

The action remains the inherited 20-dimensional joint-target action.

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
.pt checkpoint ─────┼─ evaluate_rsl.py → robust metrics
                    └─ play_rsl.py     → raw MP4 + overlay CSV
```
