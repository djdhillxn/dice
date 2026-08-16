"""Utilities for keeping each DICE run self-contained and easy to inspect."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


def export_tensorboard_scalars(run_dir: Path) -> dict:
    """Export TensorBoard scalars to a portable CSV and compact JSON summary."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = Path(run_dir)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return {"status": "skipped", "reason": "no TensorBoard event file found"}

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    if not tags:
        return {"status": "skipped", "reason": "no TensorBoard scalars found"}

    series = {tag: accumulator.Scalars(tag) for tag in tags}
    iteration_tags = [tag for tag in tags if not tag.endswith("/time")]
    iteration_steps = sorted(
        {event.step for tag in iteration_tags for event in series[tag]}
    )
    values_by_tag = {
        tag: {event.step: event.value for event in series[tag]}
        for tag in iteration_tags
    }

    csv_path = run_dir / "training_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", *iteration_tags])
        writer.writeheader()
        for step in iteration_steps:
            row = {"iteration": step}
            row.update(
                {
                    tag: values_by_tag[tag][step]
                    for tag in iteration_tags
                    if step in values_by_tag[tag]
                }
            )
            writer.writerow(row)

    scalar_summary = {}
    total_scalar_points = 0
    for tag, events in series.items():
        values = [event.value for event in events]
        steps = [event.step for event in events]
        total_scalar_points += len(values)
        minimum = min(values)
        maximum = max(values)
        tail = values[-min(100, len(values)) :]
        scalar_summary[tag] = {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": minimum,
            "minimum_step": steps[values.index(minimum)],
            "maximum": maximum,
            "maximum_step": steps[values.index(maximum)],
            "last_100_mean": sum(tail) / len(tail),
        }

    summary_path = run_dir / "training_summary.json"
    summary = {
        "source_event_files": [path.name for path in event_files],
        "scalar_tag_count": len(tags),
        "scalar_point_count": total_scalar_points,
        "iteration_count": len(iteration_steps),
        "scalars": scalar_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "status": "complete",
        "metrics_csv": csv_path.name,
        "summary_json": summary_path.name,
        "scalar_tag_count": len(tags),
        "scalar_point_count": total_scalar_points,
        "iteration_count": len(iteration_steps),
    }


def collect_runtime_logs(run_dir: Path, since_timestamp: float) -> dict:
    """Copy Isaac Lab and Kit logs created during this run into its directory."""

    run_dir = Path(run_dir)
    candidates = []

    isaaclab_log_root = Path("/tmp/isaaclab/logs")
    if isaaclab_log_root.is_dir():
        candidates.extend(isaaclab_log_root.glob("isaaclab_*.log"))

    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    kit_log_root = site_packages / "isaacsim" / "kit" / "logs"
    if kit_log_root.is_dir():
        candidates.extend(kit_log_root.rglob("kit_*.log"))

    selected = []
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_mtime >= since_timestamp - 120.0:
                selected.append(path)
        except OSError:
            continue

    if not selected:
        return {"status": "skipped", "reason": "no run-time Isaac logs found"}

    destination = run_dir / "runtime_logs"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(set(selected)):
        target = destination / source.name
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        copied.append(str(target.relative_to(run_dir)))

    return {
        "status": "complete" if copied else "skipped",
        "files": copied,
    }


def write_artifact_manifest(run_dir: Path) -> dict:
    """Write a size manifest for every other artifact in a run directory."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "artifact_manifest.json"
    files = []
    total_bytes = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(path.relative_to(run_dir)), "bytes": size})

    manifest = {
        "run_directory": str(run_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "path": manifest_path.name,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
