"""Checkpoint discovery and deterministic evaluation-ranking helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


_NUMBERED_CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")

RANKING_PRIORITY = (
    "meets_command_success_threshold (descending)",
    "drop_rate (ascending)",
    "mean_consecutive_commands (descending)",
    "minimum_per_face_success_rate (descending)",
    "median_time_to_target_seconds (ascending)",
    "deterministic_action_out_of_bounds_fraction (ascending)",
)


def resolve_run_directory(run_reference, outputs_root):
    """Resolve either a run directory path or an ID below ``outputs_root``."""

    reference = Path(run_reference).expanduser()
    candidates = [reference, Path(outputs_root).expanduser() / reference]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run directory not found. Checked: {checked}")


def discover_checkpoints(run_directory, include_initial=False):
    """Return numbered checkpoints in iteration order followed by ``model_final``."""

    run_directory = Path(run_directory)
    numbered = []
    for checkpoint in run_directory.glob("model_*.pt"):
        match = _NUMBERED_CHECKPOINT.fullmatch(checkpoint.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration == 0 and not include_initial:
            continue
        numbered.append((iteration, checkpoint.resolve()))

    checkpoints = [path for _, path in sorted(numbered)]
    final_checkpoint = run_directory / "model_final.pt"
    if final_checkpoint.is_file():
        checkpoints.append(final_checkpoint.resolve())

    if not checkpoints:
        message = f"No eligible model_*.pt checkpoints found in {run_directory}"
        if not include_initial:
            message += " (model_0.pt is excluded by default)"
        raise FileNotFoundError(message)
    return checkpoints


def evaluation_is_reusable(summary_path, checkpoint, task, seed, episodes, num_envs):
    """Return whether an existing summary matches the requested evaluation."""

    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        recorded_checkpoint = Path(summary["checkpoint"]).name
        return (
            recorded_checkpoint == Path(checkpoint).name
            and summary["task"] == task
            and int(summary["seed"]) == int(seed)
            and int(summary["episodes"]) == int(episodes)
            and int(summary["num_envs"]) == int(num_envs)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def ranking_entry(summary_path):
    """Convert one evaluator summary into the checkpoint-selection fields."""

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    per_face_rates = {
        str(face): float(summary["per_face"][str(face)]["success_rate"])
        for face in range(1, 7)
    }
    latency = summary.get("median_time_to_target_seconds")
    return {
        "checkpoint": Path(summary["checkpoint"]).name,
        "summary": str(summary_path),
        "issued_command_completion_rate": float(
            summary["issued_command_completion_rate"]
        ),
        "drop_rate": float(summary["drop_rate"]),
        "mean_consecutive_commands": float(summary["mean_consecutive_commands"]),
        "minimum_per_face_success_rate": min(per_face_rates.values()),
        "minimum_per_face": min(per_face_rates, key=per_face_rates.get),
        "median_time_to_target_seconds": (
            float(latency) if latency is not None else None
        ),
        "deterministic_action_out_of_bounds_fraction": float(
            summary["deterministic_action_out_of_bounds_fraction"]
        ),
        "per_face_success_rate": per_face_rates,
    }


def rank_entries(entries, success_threshold=0.90):
    """Rank checkpoint entries using the documented lexicographic priorities."""

    ranked = []
    for entry in entries:
        item = dict(entry)
        item["meets_command_success_threshold"] = (
            item["issued_command_completion_rate"] >= success_threshold
        )
        ranked.append(item)

    def sort_key(item):
        latency = item["median_time_to_target_seconds"]
        return (
            not item["meets_command_success_threshold"],
            item["drop_rate"],
            -item["mean_consecutive_commands"],
            -item["minimum_per_face_success_rate"],
            float("inf") if latency is None else latency,
            item["deterministic_action_out_of_bounds_fraction"],
            item["checkpoint"],
        )

    ranked.sort(key=sort_key)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def format_ranking_table(ranked):
    """Render a compact terminal table for ranked checkpoints."""

    header = (
        f"{'Rank':>4}  {'Checkpoint':<16} {'Pass?':>6} {'Success':>9} "
        f"{'Drop':>9} {'MeanCmd':>9} {'MinFace':>11} "
        f"{'Latency':>9} {'OOB':>9}"
    )
    lines = [header, "-" * len(header)]
    for item in ranked:
        latency = item["median_time_to_target_seconds"]
        latency_text = "n/a" if latency is None else f"{latency:.3f}s"
        face_text = (
            f"{100.0 * item['minimum_per_face_success_rate']:.2f}%"
            f"(F{item['minimum_per_face']})"
        )
        lines.append(
            f"{item['rank']:>4}  {item['checkpoint']:<16} "
            f"{str(item['meets_command_success_threshold']):>6} "
            f"{100.0 * item['issued_command_completion_rate']:>8.2f}% "
            f"{100.0 * item['drop_rate']:>8.2f}% "
            f"{item['mean_consecutive_commands']:>9.3f} "
            f"{face_text:>11} {latency_text:>9} "
            f"{100.0 * item['deterministic_action_out_of_bounds_fraction']:>8.2f}%"
        )
    return "\n".join(lines)


def write_ranking_artifacts(summary_paths, output_directory, success_threshold=0.90):
    """Rank summaries and write JSON, CSV, and text selection artifacts."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    ranked = rank_entries(
        [ranking_entry(path) for path in summary_paths],
        success_threshold=success_threshold,
    )
    table = format_ranking_table(ranked)
    payload = {
        "project": "DICE",
        "success_threshold": success_threshold,
        "ranking_priority": list(RANKING_PRIORITY),
        "evaluated_checkpoint_count": len(ranked),
        "selected_checkpoint": ranked[0]["checkpoint"],
        "ranked_checkpoints": ranked,
    }
    (output_directory / "ranking.json").write_text(json.dumps(payload, indent=2))
    (output_directory / "ranking.txt").write_text(table + "\n")
    (output_directory / "selected_checkpoint.txt").write_text(
        ranked[0]["checkpoint"] + "\n"
    )

    csv_fields = [
        "rank",
        "checkpoint",
        "meets_command_success_threshold",
        "issued_command_completion_rate",
        "drop_rate",
        "mean_consecutive_commands",
        "minimum_per_face_success_rate",
        "minimum_per_face",
        "median_time_to_target_seconds",
        "deterministic_action_out_of_bounds_fraction",
        "summary",
    ]
    with (output_directory / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    return payload, table
