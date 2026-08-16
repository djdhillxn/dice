# DICE results template

## System

- Isaac Lab / Isaac Sim environment:
- GPU:
- number of parallel environments:
- control timestep:
- training iterations:
- selected checkpoint:

## Training configuration

- actor observation dimension: 126
- critic state dimension: 247
- action dimension: 20
- steps per environment: 32
- actor/critic architecture: 512, 512, 256, 128
- PPO clip: 0.2
- learning rate: 3e-4, fixed
- desired KL: inactive under fixed schedule
- epochs / minibatches: 5 / 4
- gamma / lambda: 0.99 / 0.95
- raw/applied action boundary: unbounded Gaussian / environment clamp to [-1, 1]
- raw-action boundary penalty: -0.1 beyond |a| = 0.9, squared
- global reward scale: 0.1

## Final task definition

- success angle: 16 degrees
- hold duration: 20 control steps
- angular-speed threshold: 1.25 rad/s
- position tolerance: 0.12 m
- command sequence behavior: new different face without object reset

## Evaluation

| Metric | Nominal | Symmetric robust | Adverse material |
|---|---:|---:|---:|
| Episodes | | | |
| Command success rate | | | |
| Drop rate | | | |
| Mean commands per episode | | | |
| Median commands per episode | | | |
| Commands per simulated minute | | | |
| Median command latency | | | |
| Deterministic action OOB rate | | | |

## Per-face success

| Face | Nominal | Symmetric robust | Adverse material |
|---|---:|---:|---:|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |

Physics conditions:

- nominal: no evaluation events
- symmetric robust: object mass/friction samples in `[0.8, 1.2]`, constrained
  so dynamic friction does not exceed static friction
- adverse material: fixed `1.5x` object mass and `0.7` static/dynamic friction

## Qualitative result

- strongest behavior:
- most common failure:
- whether failure is orientation, retention, settling, or dropping:
- selected video command sequence:
