"""Plot the compact task-metric CSV produced during training."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description="Plot DiceDial training metrics.")
parser.add_argument("--csv", required=True)
parser.add_argument("--output", default="training_metrics.png")
args = parser.parse_args()

frame = pd.read_csv(args.csv)
metrics = [
    "alignment",
    "hold_progress",
    "commands_completed",
    "out_of_reach",
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
for axis, metric in zip(axes, metrics):
    if metric not in frame:
        continue
    axis.plot(frame["timesteps"], frame[metric])
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.grid(True, alpha=0.25)
axes[-1].set_xlabel("Environment timesteps")
fig.tight_layout()
fig.savefig(args.output, dpi=180)
print("Saved", Path(args.output).resolve())
