# DICE final results

## Closure decision

The training and quantitative evaluation phases are complete. The selected
policy demonstrates high-throughput continuous die reorientation under nominal
physics, retains that behavior under moderate held-out material variation, and
exposes a clear grasp-retention boundary under a fixed heavy, low-friction
stress condition. No additional training or evaluation run is required to
support the present project scope.

The result supports a claim of **simulation and material-parameter robustness**.
It does not establish real-robot performance or comprehensive sim-to-real
robustness.

## Provenance

| Item | Final value |
|---|---|
| Training run | `2026-08-16_11-22-43_angular_bound_pilot_gurgaon` |
| Training seed | 42 |
| Training environments | 2,048 |
| Rollout steps / environment | 32 |
| PPO iterations | 5,000 |
| Training transitions | 327,680,000 |
| Selected checkpoint | `model_4000.pt` |
| Checkpoint SHA-256 | `16261be09bc3ca57b40fdf2a715216c76298d8b377ef280fadc06ba6ad6a58af` |
| Checkpoint candidates evaluated | 5 |
| Final episodes | 1,000 per condition |
| Evaluation environments | 256 concurrent simulator instances |
| Hardware | NVIDIA L4, 23.7 GB GPU memory |
| Software | Isaac Lab 2.3.2.post1, RSL-RL 3.1.2, PyTorch 2.7.0+cu128 |

The 256 evaluation environments control only how many simulator instances run
concurrently. They do not weaken the task, shorten episodes, or reduce the
1,000-episode sample. Training used 2,048 environments because collecting PPO
rollouts efficiently has a different memory/throughput trade-off from loading
one frozen policy for deterministic evaluation.

`model_4000.pt` was selected before final testing. In the 500-episode
checkpoint sweep it achieved a 97.10% issued-command completion rate, 9.40%
drop rate, and 33.424 mean completed commands per episode. The later
`model_final.pt` was faster by 0.033 seconds in median command latency but had a
higher 14.00% drop rate, so it was not selected.

## Final evaluation

All three evaluations used the identical selected checkpoint. The condition
seeds were 2026 (nominal), 2027 (symmetric physics variation), and 2028
(adverse stress).

| Metric | Nominal | Symmetric physics variation | Heavy / low-friction stress |
|---|---:|---:|---:|
| Episodes | 1,000 | 1,000 | 1,000 |
| Successful / issued commands | 33,334 / 34,334 | 33,072 / 34,072 | 23,514 / 24,514 |
| Issued-command completion | 97.09% | 97.07% | 95.92% |
| Drop rate | 9.70% | 9.50% | 45.30% |
| Approx. 95% interval for drop rate | 8.02–11.69% | 7.84–11.48% | 42.24–48.40% |
| Mean completed commands / episode | 33.334 | 33.072 | 23.514 |
| Median completed commands / episode | 37 | 37 | 32 |
| Commands / simulated minute | 90.536 | 89.000 | 81.996 |
| Median command latency | 0.617 s | 0.617 s | 0.650 s |
| Episodes completing at least one command | 97.40% | 98.10% | 96.70% |
| Minimum per-face completion | 96.88% (face 3) | 96.80% (face 5) | 95.11% (face 6) |
| Deterministic action OOB rate | 20.77% | 20.77% | 20.91% |

The drop-rate intervals are Wilson binomial intervals over episodes. They
describe evaluation sampling uncertainty only; they do not account for the
single training seed or simulator-model uncertainty.

## How to interpret command completion

The policy receives a new, different face command immediately after every
completion. At episode termination, exactly one command remains active and is
counted as unfinished. Consequently,

\[
\text{issued-command completion}
=
\frac{\text{completed commands}}
{\text{completed commands} + \text{episodes}}.
\]

For the adverse condition this is `23,514 / 24,514 = 95.92%`. It means the
policy completed many sequential commands before each episode ended; it is not
a one-shot probability that the object will never be dropped. Drop rate, mean
commands per episode, and latency provide the necessary complementary view.

## Findings

### 1. Nominal manipulation is fast and balanced

The policy completed a mean of 33.334 commands per 24-second episode with a
0.617-second median latency. All six faces exceeded 96.88% issued-command
completion, and 97.4% of episodes completed at least one command. The 9.7%
episode drop rate is the observed safety/throughput trade-off of this aggressive
continuous policy.

### 2. Moderate held-out physics variation caused no measurable degradation

The symmetric condition sampled object mass and material coefficients within
`[0.8, 1.2]` of nominal. Relative to nominal, its drop rate changed by only
`-0.2` percentage points and mean commands by `-0.262`. Approximate 95%
intervals for both differences include zero. Latency, action OOB rate, median
commands, and minimum-face performance were effectively unchanged.

This condition is best described as **symmetric held-out physics variation**.
Robustness is the observed retention of performance across that variation, not
an assertion that every sampled instance is harder than nominal.

### 3. The adverse corner revealed a retention boundary

The fixed `1.5x` object mass and `0.7` static/dynamic object friction increased
the drop rate by 35.6 percentage points, from 9.7% to 45.3%. The approximate
95% interval for that increase is 32.0–39.2 percentage points. Mean completed
commands fell by 9.82 (29.5%), while median latency rose by only 0.033 seconds
and action OOB rate by only 0.14 percentage points.

The episode records show that this was mainly a long-horizon retention failure:

| Adverse drop decomposition | Value |
|---|---:|
| Dropped episodes | 453 / 1,000 |
| Drops after at least 1 completed command | 424 |
| Drops after at least 10 completed commands | 219 |
| Drops after at least 20 completed commands | 109 |
| Drops after at least 30 completed commands | 40 |
| Mean commands before a drop | 11.49 |
| Median commands before a drop | 9 |
| Median time before a drop | 7.28 s |
| Mean commands in adverse episodes surviving to timeout | 33.47 |

Thus, the adverse policy generally retained semantic targeting and rapid
reorientation, but the probability of eventually losing the die accumulated
over a long command sequence. A heavier object requires greater supporting and
rotational contact forces, while lower friction reduces the tangential-force
margin before slip. Their combination is a plausible mechanism for the
observed retention loss. Because both parameters changed together, this
experiment does not identify their separate causal contributions.

The configured coefficients belong to the object material. PhysX combines the
materials on both contacting shapes; under the configured `average` combine
mode, the effective pair coefficient is `(a + b) / 2`. The report therefore
does not describe `0.7` as a guaranteed 30% reduction in every effective
fingertip contact.

## Limitations

- One PPO training seed was run. The 1,000-episode intervals do not measure
  training-seed variability.
- Each final condition used one fixed evaluation seed. The conditions are
  reproducible but not paired trajectory-by-trajectory.
- The adverse condition changes mass and friction jointly; it is a stress test,
  not a mass-versus-friction ablation.
- Robustness covers object material parameters only. Observation noise,
  control latency, actuator error, contact compliance, geometry variation, and
  real hardware were not evaluated.
- Issued-command completion is coupled to sequential throughput and must not be
  presented as an independent one-shot success probability.
- The deterministic action OOB metric is diagnostic of the Gaussian actor mean;
  the environment still clamps commands before applying them to the hand.

## Future work

If the project is extended, the highest-value next step is training-time domain
randomization over plausible mass, friction, actuator, and latency ranges, then
reevaluating the unchanged adverse corner. Separate mass-only and friction-only
ablations would identify the dominant failure mechanism. Multi-seed training
and real-object system identification would be required before making a broad
sim-to-real claim.

Those are extensions rather than missing evidence for the present goal. The
current experiment already establishes a successful nominal policy, measured
moderate material robustness, and an honest, reproducible failure boundary.

## Primary artifacts

On the evaluation host, the complete machine-readable record is stored under:

```text
outputs/2026-08-16_11-22-43_angular_bound_pilot_gurgaon/
├── run.json
├── training_metrics.csv
├── training_summary.json
├── model_4000.pt
└── evaluation/
    ├── checkpoint_sweep/
    └── final_model_4000/
        ├── nominal/{episodes.csv,summary.json}
        ├── robust/{episodes.csv,summary.json}
        ├── adverse/{episodes.csv,summary.json}
        ├── evaluation_run.json
        ├── final_comparison.csv
        ├── final_comparison.txt
        └── final_summary.json
```

Physics interpretation follows the NVIDIA PhysX documentation for
[rigid-body mass and Coulomb friction](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/RigidBodyDynamics.html)
and the documented
[material combine modes](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html).

A compact, tracked copy of the comparison table is available as
[`final_comparison.csv`](final_comparison.csv). The larger checkpoints and
per-episode files remain under the ignored `outputs/` run directory.
