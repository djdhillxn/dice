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

## Completed primary run

The final full-task PPO experiment completed with:

```text
seed: 42
environments: 2048
steps per environment: 32
maximum iterations: 5000
checkpoint interval: 1000
completed transitions: 327680000
selected checkpoint: model_4000.pt
```

No success thresholds, rewards, object distribution, or command distribution change during training.

## Checkpoint selection

The run already produces regular checkpoints. Use nominal frozen-policy evaluation to compare late checkpoints rather than launching new training configurations immediately.

Rank candidate checkpoints by:

1. whether command completion meets the 90% acceptance threshold
2. drop rate
3. mean completed commands per episode
4. minimum per-face success rate
5. median successful-command latency
6. deterministic action out-of-bounds rate

Training reward is diagnostic, not the final selection metric.

Run the nominal checkpoint sweep with:

```bash
python -u scripts/run_checkpoint_sweep.py <timestamp>_<run_name>
```

The sweep excludes initialization-only `model_0.pt`, evaluates periodic
checkpoints plus `model_final.pt` sequentially, and writes isolated summaries
and a ranking under `evaluation/checkpoint_sweep/`. The automated ranking first
separates checkpoints that meet the 90% command-success threshold, then applies
drop rate, mean commands, minimum per-face success, latency, and action-bound
rate as ordered tie-breakers.

## Final reporting

For the selected checkpoint, report:

- 1,000 nominal episodes (`seed = 2026`)
- 1,000 symmetric held-out mass/friction episodes (`seed = 2027`)
- 1,000 fixed adverse-material episodes (`seed = 2028`)
- command success rate
- drop rate
- mean and median completed commands per episode
- completed commands per simulated minute
- median time to successful command
- deterministic action out-of-bounds rate
- six per-face success rates

The symmetric robustness condition samples object mass and the two friction
coefficients within 0.8 to 1.2 times nominal, then constrains dynamic friction
to be no greater than static friction. The adverse condition fixes mass at 1.5
times nominal and both friction coefficients at 0.7. It is a deliberately
difficult material stress test, not another symmetric distribution. All three
conditions must remain
separate because they use different physics distributions.

The final evaluation completed all three 1,000-episode conditions from the
same selected checkpoint. Nominal and symmetric-variation drop rates were 9.7%
and 9.5%, respectively; adverse heavy/low-friction evaluation produced a 45.3%
drop rate while still completing a mean of 23.514 sequential commands per
episode. The experiment and its quantitative evaluation are closed. See
[final_results.md](final_results.md) for the full analysis.

## Presentation

The final presentation package contains three orthogonal videos: synchronized
oblique/top nominal success, nominal versus held-out ±20% physics variation,
and synchronized oblique/side adverse retention failure. All policy footage is
shown at 0.5× playback, without static title/result cards, and each export has
a copy-ready Markdown companion. It is generated from the frozen
`model_4000.pt` checkpoint by `scripts/render_portfolio_videos.py`. Rendering
does not alter or extend the completed quantitative evaluation. One fixed seed
(`9`) is used consistently for the final representative videos; it is not the
source of the aggregate performance claims. Nominal and symmetric variation run
for 12 confirmed commands, while the adverse story continues to its drop. Physics audit,
trajectory replay, encoding, and GitHub Pages requirements are defined in
[video_rendering.md](video_rendering.md).
