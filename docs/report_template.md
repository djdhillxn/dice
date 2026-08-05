# DICE results template

## System

- Isaac Lab / Isaac Sim environment:
- GPU:
- number of parallel environments:
- control timestep:
- training iterations:
- selected checkpoint:

## Training configuration

- observation dimension: 165
- action dimension: 20
- steps per environment: 16
- actor/critic architecture: 512, 512, 256, 128
- PPO clip: 0.2
- learning rate: 5e-4, adaptive
- desired KL: 0.016
- epochs / minibatches: 5 / 4
- gamma / lambda: 0.99 / 0.95

## Final task definition

- success angle: 16 degrees
- hold duration: 20 control steps
- angular-speed threshold: 1.25 rad/s
- position tolerance: 0.12 m
- command sequence behavior: new different face without object reset

## Evaluation

| Metric | Nominal | Robust |
|---|---:|---:|
| Episodes | | |
| Command success rate | | |
| Drop rate | | |
| Mean commands per episode | | |
| Median commands per episode | | |
| Median command latency | | |

## Per-face success

| Face | Nominal | Robust |
|---|---:|---:|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |

## Qualitative result

- strongest behavior:
- most common failure:
- whether failure is orientation, retention, settling, or dropping:
- selected video command sequence:
