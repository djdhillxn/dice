# DICE architecture

## Simulator layer

DICE subclasses Isaac Lab's direct in-hand manipulation environment. Isaac Lab retains responsibility for the Shadow Hand articulation, contact simulation, stock DexCube, vectorized resets, fingertip state, action smoothing, and joint-target application.

The custom environment adds only semantic face commands, yaw-invariant face alignment, hold-to-confirm completion, command switching, reward terms, and task diagnostics.

## Training object versus presentation object

Training, nominal evaluation, and robust evaluation use Isaac Lab's stock instanceable DexCube. This is the performance-sensitive and known-good object path.

The single-environment play configuration replaces it with the local numbered die. Its collision size and density are aligned with the stock cube configuration, but its visible pips are used only for the final video.

## Policy interface

The actor receives 165 values:

```text
157 stock Shadow Hand full-observation features
  6 requested-face one-hot values
  1 normalized hold counter
  1 requested-face alignment
---
165 total
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
