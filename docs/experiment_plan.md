# DICE experiment plan

## Simulator contract preflight

Before every full run following an observation, action, controller, or critic
change, execute:

```bash
bash scripts/run_training_preflight.sh 64 2
```

Proceed only if reset reports policy shape `[64, 126]`, critic shape
`[64, 247]`, and both PPO iterations complete. Check the logged fingertip-load
proxy saturation; a value near `1.0` means `fingertip_load_scale` must be
reduced before the full run.

## Primary run

Run one full-task PPO experiment:

```text
seed: 42
environments: 2048
steps per environment: 32
maximum iterations: 10000
checkpoint interval: 1000
```

No success thresholds, rewards, object distribution, or command distribution change during training.

## Checkpoint selection

The run already produces regular checkpoints. Use nominal frozen-policy evaluation to compare late checkpoints rather than launching new training configurations immediately.

Rank candidate checkpoints by:

1. mean completed commands per episode
2. drop rate
3. minimum per-face success rate
4. median successful-command latency

Training reward is diagnostic, not the final selection metric.

## Final reporting

For the selected checkpoint, report:

- 500 nominal episodes
- 500 held-out robustness episodes
- command success rate
- drop rate
- mean and median completed commands per episode
- median time to successful command
- six per-face success rates

The robustness numbers must remain separate from nominal performance because the environments use different physics distributions.

## Presentation

Render one deterministic six-command episode with the numbered die and no randomization. Keep the raw video and the annotated version.
