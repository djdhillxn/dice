# Experiment plan

## Development gates

### Gate 0 — task contract

Run `scripts/smoke_test.py` with 16 environments. Require finite observations and rewards, the expected `(num_envs, 164)` policy observation, and valid face commands.

### Gate 1 — face-one retention

Train `DiceDial-Shadow-Easy-v0`. Inspect alignment, hold progress, and drop rate. The purpose is to verify the reward, object asset, and policy/controller interface before testing command conditioning.

### Gate 2 — six-face command conditioning

Warm-start `DiceDial-Shadow-Random-v0` from Gate 1. Evaluate per-face success counts to ensure the policy is not ignoring the one-hot command or collapsing to one easy face.

### Gate 3 — continuous command sequence

Warm-start `DiceDial-Shadow-Sequence-v0` from Gate 2. The main metric becomes consecutive commands before a drop or timeout.

## Final evaluation

Use a fixed held-out seed and at least 500 completed episodes. Report:

- target-face success rate
- median time to target
- die-drop rate
- mean and maximum consecutive commands

Run three seeds for the final portfolio result. Aggregate mean and standard deviation across seeds; preserve each seed's raw `episodes.csv` and `summary.json`.

## Optional robustness pass

After the nominal sequence policy works, evaluate the same frozen checkpoint on `DiceDial-Shadow-Robust-v0`. This task uses Isaac Lab event terms to scale die mass by 0.8–1.2 and randomize static/dynamic friction over 0.8–1.2. Keep nominal and randomized tables separate so robustness does not obscure basic task competence.
