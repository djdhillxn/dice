# DiceDial project report template

## Project question

Can one command-conditioned dexterous-hand policy expose any requested die face and execute multiple face commands without releasing the object?

## Method

Describe the inherited Isaac Lab Shadow Hand environment, the semantic face command, the yaw-invariant alignment objective, the hold condition, and the SB3 PPO configuration.

## Experimental protocol

Document simulator and package versions, GPU, environment count, total environment steps, random seeds, curriculum checkpoints, and deterministic evaluation settings.

## Results

| Seed | Face success | Median target time | Drop rate | Mean sequence | Max sequence |
|---|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## Ablation

Compare the full hold-to-confirm objective against an alignment-only reward. The intended question is whether explicit settling and continuous hold improve stable command completion rather than merely producing transient face crossings.

## Qualitative analysis

Include successful transitions, slow transitions, oscillatory near-successes, command-ignoring behavior, and drops. Link the annotated video and note whether failures are caused by contact loss, insufficient rotation, or inability to settle.

## Limitations

State that the policy uses privileged simulator state, remains simulation-only, and has not demonstrated sim-to-real transfer. Avoid claiming general dexterous manipulation from one die task.
