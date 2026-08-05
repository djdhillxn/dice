"""Aggregate multiple DiceDial evaluation summaries across held-out seeds."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description="Aggregate DiceDial summary.json files.")
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", default="evaluation/aggregate")
args = parser.parse_args()


HEADLINE_FIELDS = [
    "target_face_success_rate",
    "median_time_to_target_seconds",
    "drop_rate",
    "mean_consecutive_commands",
    "median_consecutive_commands",
    "max_consecutive_commands",
]


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "count": int(array.size),
    }


summaries = []
for pattern in args.inputs:
    matches = [Path(item) for item in sorted(glob.glob(pattern))] if any(char in pattern for char in "*?[]") else [Path(pattern)]
    for path in matches:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        payload["source"] = str(path)
        summaries.append(payload)

if not summaries:
    raise RuntimeError("No evaluation summaries were found.")

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
for summary in summaries:
    row = {"source": summary["source"], "task": summary["task"], "seed": summary["seed"]}
    for field in HEADLINE_FIELDS:
        row[field] = summary.get(field)
    for face in range(1, 7):
        row[f"face_{face}_success_rate"] = summary["per_face"][str(face)]["success_rate"]
    rows.append(row)

frame = pd.DataFrame(rows)
frame.to_csv(output_dir / "runs.csv", index=False)

aggregate = {
    "task_values": sorted(frame["task"].unique().tolist()),
    "seeds": [int(seed) for seed in frame["seed"].tolist()],
    "num_runs": int(len(frame)),
    "headline": {},
    "per_face_success_rate": {},
}
for field in HEADLINE_FIELDS:
    values = frame[field].dropna().tolist()
    aggregate["headline"][field] = _stats(values) if values else None
for face in range(1, 7):
    aggregate["per_face_success_rate"][str(face)] = _stats(frame[f"face_{face}_success_rate"].tolist())

(output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
print(json.dumps(aggregate, indent=2))
