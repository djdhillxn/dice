# DICE Dial GCE and runtime setup

This document keeps VM- and environment-specific setup out of the main README. It records the compatibility fixes used for the final DICE Dial run on Ubuntu 22.04 / Google Compute Engine.

## Activate and verify the environment

The final workflow used a Conda environment named `dice`:

```bash
conda activate dice
cd ~/projects/dice
python --version
which python
```

The repository does not currently ship an `environment.yml`; project-level Python dependencies are defined in `pyproject.toml`.

```bash
python -m pip install -e .
```

For the video pipeline:

```bash
python -m pip install -e ".[video]"
```

The final Isaac Sim 5.1 environment uses:

- `numpy==1.26.0`
- `opencv-python-headless==4.11.0.86`

The NumPy pin avoids silently upgrading the Isaac runtime to an unsupported NumPy 2.x combination. The headless OpenCV wheel avoids unnecessary GUI dependencies on the VM.

## Repair an accidental NumPy / OpenCV upgrade

If a previous install upgraded NumPy to 2.x or installed a conflicting OpenCV wheel:

```bash
python -m pip uninstall -y \
  opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless

python -m pip install --upgrade \
  "numpy==1.26.0" \
  "opencv-python-headless==4.11.0.86"

python -m pip install -e ".[video]"
python -m pip check
```

Verify the active runtime:

```bash
python - <<'PY'
import cv2
import numpy
import sqlite3

print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("sqlite:", sqlite3.sqlite_version)
PY
```

`scripts/train_rsl.py` also checks the NumPy version before launching Isaac Sim so an incompatible environment fails early.

## `CXXABI_1.3.15` / `libstdc++.so.6` on Ubuntu 22.04

If Isaac Sim reports that `/lib/x86_64-linux-gnu/libstdc++.so.6` does not provide `CXXABI_1.3.15`, inspect the Conda runtime first:

```bash
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep CXXABI_1.3.15
```

If the symbol is missing:

```bash
conda install -y -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
```

Then prefer the active environment's libraries in the current shell and verify `sqlite3` before launching Isaac Sim:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import sqlite3; print('sqlite OK:', sqlite3.sqlite_version)"
```

## Training preflight

After changing the actor/critic observation contract, action path, or controller, run the short simulator contract preflight before a paid full job:

```bash
bash scripts/run_training_preflight.sh 64 2
```

A successful preflight must report policy shape `[64, 126]`, critic shape `[64, 247]`, complete both PPO iterations, and write a run with `"status": "complete"` under `outputs/preflight/`.

The repeated headless message

```text
sh: 1: zenity: not found
```

is not a request to install `zenity`. Use `--headless` on the VM.

## Full headless training

```bash
python -u scripts/train_rsl.py \
  --task DICE-Shadow-Train-v0 \
  --num_envs 2048 \
  --max_iterations 5000 \
  --run_name angular_bound_pilot_gurgaon \
  --headless
```

`python -u` keeps terminal output unbuffered over SSH. The launcher persists startup milestones in `outputs/<run>/startup.log`, which is the first place to inspect if Gym creation, wrapper reset, runner construction, or PPO startup stalls.

## TensorBoard

Each full run stores the native RSL-RL TensorBoard event file plus a portable `training_metrics.csv` and `training_summary.json`.

```bash
tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

Forward port `6006` over SSH and open `http://localhost:6006` locally.

## Portfolio video dependencies

The end-to-end presentation renderer requires FFmpeg/FFprobe and DejaVu fonts:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core

ffmpeg -hide_banner -encoders | grep -E 'libx264|libvpx-vp9'
ffprobe -version
```

Then:

```bash
python -m pip install -e ".[video]"
```

See [`video_rendering.md`](video_rendering.md) for the rendering contract.

## Conda YAML, if one is added later

If the repository later adds an `environment.yml`:

```bash
conda env create -n dice -f environment.yml
conda activate dice
python -m pip install -e ".[video]"
```

For updates:

```bash
conda env update -n dice -f environment.yml --prune
conda activate dice
python -m pip install -e ".[video]"
```

Changes only to `pyproject.toml` do not require recreating the Conda environment; rerun the editable install.

## GCE lifecycle note

A `tmux` session survives an SSH disconnect, but it does **not** survive stopping the Compute Engine VM. Stopping the instance terminates the running training/rendering processes and releases the accelerator until the VM is started again. Use `tmux` to protect against network/session loss, and stop the VM only after the desired job has completed or its outputs have been safely persisted.
