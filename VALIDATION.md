# Validation status

Validated in the artifact-building environment on August 5, 2026:

- all Python files compile with `compileall`
- six CPU-only tests pass (four PyTorch geometry tests and two USD text-structure tests)
- canonical quaternions place each of the six requested faces upward
- opposite-face convention is consistent (`1–6`, `2–5`, `3–4`)
- repository packaging includes YAML and USD package data

Not executable in the artifact-building environment:

- Isaac Sim / Isaac Lab launch
- PhysX contact simulation
- RTX video rendering
- USD parsing through `pxr`
- end-to-end PPO training

Reason: the available runtime uses Python 3.13 and has no Isaac Sim, Isaac Lab, `pxr`, or compatible NVIDIA simulation stack. The repository intentionally targets Python 3.11 for Isaac Sim 5.1. Run `scripts/smoke_test.py` as the first full-stack gate on the target machine.
