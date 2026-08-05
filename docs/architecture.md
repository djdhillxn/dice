# Architecture

## Reused components

DiceDial deliberately reuses Isaac Lab's direct Shadow Hand cube-reorientation stack:

- Shadow Hand articulation and its 20-dimensional joint-target controller
- fingertip rigid-body state extraction
- GPU-vectorized environment cloning
- object and hand reset routines
- action smoothing and joint-limit scaling
- the original 157-dimensional full proprioceptive observation
- Isaac Lab's `Sb3VecEnvWrapper`
- Stable-Baselines3 PPO, callbacks, TensorBoard logging, and `VecNormalize`
- Gymnasium's `RecordVideo`

## Custom components

The project adds a small semantic task layer:

1. A die convention with local face normals for numbers 1–6.
2. A six-dimensional one-hot command and scalar hold-progress observation.
3. A face-up alignment reward invariant to yaw.
4. A continuous hold condition based on alignment, palm distance, and angular speed.
5. Command switching after a held success, without resetting the die.
6. Evaluation metrics and a video overlay.
7. A visual USD die with colored numbered faces and a single cube collider.

## Observation

The policy receives 164 values:

- 157 values from the inherited Shadow Hand full observation
- 6 values for the requested die face
- 1 value for normalized hold progress

The inherited observation already includes hand joints, hand velocities, die pose and velocity, fingertip poses and velocities, goal orientation, and previous actions.

## Reward

The reward is:

```text
alignment reward
+ palm-retention term
+ angular-speed term
+ action regularization
+ correct-top-face shaping
+ hold-progress bonus
+ one-time command-completion bonus
```

The principal signal uses the dot product between the commanded local face normal, rotated into world coordinates, and world up. This means every yaw angle is acceptable and avoids over-constraining the die to one quaternion.

## Episode logic

A command succeeds only when all three conditions remain true for `hold_steps` consecutive control steps:

- face alignment exceeds the configured angular threshold
- the die remains within the palm-distance threshold
- angular speed remains below the settling threshold

After success, the sequence task samples a different face and keeps the current hand and die state. A drop or time limit ends the episode.
