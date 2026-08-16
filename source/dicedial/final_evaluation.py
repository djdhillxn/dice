"""Final three-condition evaluation definitions and artifact helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


FINAL_CONDITIONS = (
    {
        "key": "nominal",
        "label": "Nominal",
        "task": "DICE-Shadow-Eval-v0",
        "seed": 2026,
        "physics": {
            "object_mass_scale": 1.0,
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "description": "Nominal training physics; no evaluation events.",
        },
    },
    {
        "key": "robust",
        "label": "Symmetric robust",
        "task": "DICE-Shadow-Robust-v0",
        "seed": 2027,
        "physics": {
            "object_mass_scale": {"distribution": "uniform", "range": [0.8, 1.2]},
            "static_friction": {"distribution": "uniform", "range": [0.8, 1.2]},
            "dynamic_friction": {
                "source_distribution": "uniform",
                "source_range": [0.8, 1.2],
                "constraint": "dynamic_friction <= static_friction",
            },
            "make_consistent": True,
            "description": (
                "Held-out mass/material samples around nominal; material samples "
                "are constrained so dynamic friction does not exceed static friction."
            ),
        },
    },
    {
        "key": "adverse",
        "label": "Adverse heavy/slippery",
        "task": "DICE-Shadow-Adverse-v0",
        "seed": 2028,
        "physics": {
            "object_mass_scale": 1.5,
            "static_friction": 0.7,
            "dynamic_friction": 0.7,
            "make_consistent": True,
            "description": "Fixed adverse material corner; not a symmetric distribution.",
        },
    },
)

COMPARISON_FIELDS = (
    "issued_command_completion_rate",
    "drop_rate",
    "mean_consecutive_commands",
    "median_consecutive_commands",
    "completed_commands_per_sim_minute",
    "median_time_to_target_seconds",
    "minimum_per_face_success_rate",
    "episode_any_completion_fraction",
    "zero_completion_episode_fraction",
    "deterministic_action_out_of_bounds_fraction",
)


def sha256_file(path):
    """Return a streaming SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_by_key(key):
    for condition in FINAL_CONDITIONS:
        if condition["key"] == key:
            return condition
    raise KeyError(f"Unknown final-evaluation condition: {key}")


def _minimum_face(summary):
    rates = {
        str(face): float(summary["per_face"][str(face)]["success_rate"])
        for face in range(1, 7)
    }
    face = min(rates, key=rates.get)
    return face, rates[face]


def final_evaluation_is_reusable(
    summary_path, checkpoint, condition_key, episodes, num_envs
):
    """Return whether a condition summary exactly matches the requested run."""

    summary_path = Path(summary_path)
    checkpoint = Path(checkpoint)
    if not summary_path.is_file() or not checkpoint.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        condition = _condition_by_key(condition_key)
        recorded_checkpoint = summary.get("checkpoint_source", summary["checkpoint"])
        return (
            Path(recorded_checkpoint).name == checkpoint.name
            and summary["checkpoint_sha256"] == sha256_file(checkpoint)
            and summary["task"] == condition["task"]
            and int(summary["seed"]) == int(condition["seed"])
            and int(summary["episodes"]) == int(episodes)
            and int(summary["num_envs"]) == int(num_envs)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def _comparison_row(condition, summary):
    minimum_face, minimum_rate = _minimum_face(summary)
    row = {
        "condition": condition["key"],
        "label": condition["label"],
        "task": condition["task"],
        "seed": int(summary["seed"]),
        "episodes": int(summary["episodes"]),
        "successful_commands": int(summary["successful_commands"]),
        "attempted_commands": int(summary["attempted_commands"]),
        "minimum_per_face": minimum_face,
        "minimum_per_face_success_rate": minimum_rate,
    }
    for field in COMPARISON_FIELDS:
        if field == "minimum_per_face_success_rate":
            continue
        row[field] = summary.get(field)
    return row


def format_final_comparison(rows):
    """Format headline final-evaluation results for terminal and text output."""

    header = (
        f"{'Condition':<25} {'Success':>9} {'Drop':>9} {'MeanCmd':>9} "
        f"{'MinFace':>11} {'Latency':>9} {'OOB':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        latency = row["median_time_to_target_seconds"]
        latency_text = "n/a" if latency is None else f"{latency:.3f}s"
        face_text = (
            f"{100.0 * row['minimum_per_face_success_rate']:.2f}%"
            f"(F{row['minimum_per_face']})"
        )
        lines.append(
            f"{row['label']:<25} "
            f"{100.0 * row['issued_command_completion_rate']:>8.2f}% "
            f"{100.0 * row['drop_rate']:>8.2f}% "
            f"{row['mean_consecutive_commands']:>9.3f} "
            f"{face_text:>11} {latency_text:>9} "
            f"{100.0 * row['deterministic_action_out_of_bounds_fraction']:>8.2f}%"
        )
    return "\n".join(lines)


def write_final_evaluation_artifacts(summary_paths, output_directory):
    """Validate three summaries and write combined JSON/CSV/text artifacts."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries = {}
    rows = []
    for condition in FINAL_CONDITIONS:
        key = condition["key"]
        path = Path(summary_paths[key])
        summary = json.loads(path.read_text())
        if summary["task"] != condition["task"]:
            raise ValueError(
                f"{key} summary task mismatch: {summary['task']} != {condition['task']}"
            )
        if int(summary["seed"]) != int(condition["seed"]):
            raise ValueError(
                f"{key} summary seed mismatch: {summary['seed']} != {condition['seed']}"
            )
        summaries[key] = summary
        rows.append(_comparison_row(condition, summary))

    checkpoints = {
        Path(summary.get("checkpoint_source", summary["checkpoint"])).name
        for summary in summaries.values()
    }
    checkpoint_hashes = {
        summary.get("checkpoint_sha256") for summary in summaries.values()
    }
    episode_counts = {int(summary["episodes"]) for summary in summaries.values()}
    environment_counts = {int(summary["num_envs"]) for summary in summaries.values()}
    if (
        len(checkpoints) != 1
        or len(checkpoint_hashes) != 1
        or None in checkpoint_hashes
    ):
        raise ValueError("Final conditions were not evaluated from one identical checkpoint.")
    if len(episode_counts) != 1 or len(environment_counts) != 1:
        raise ValueError("Final conditions used different episode or environment counts.")

    nominal_row = rows[0]
    deltas = {}
    for row in rows[1:]:
        deltas[row["condition"]] = {
            f"{field}_delta": (
                None
                if row.get(field) is None or nominal_row.get(field) is None
                else float(row[field]) - float(nominal_row[field])
            )
            for field in COMPARISON_FIELDS
        }

    payload = {
        "schema_version": 1,
        "project": "DICE",
        "status": "complete",
        "evaluation_directory": str(output_directory),
        "checkpoint": next(iter(checkpoints)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "episodes_per_condition": next(iter(episode_counts)),
        "num_envs": next(iter(environment_counts)),
        "condition_order": [condition["key"] for condition in FINAL_CONDITIONS],
        "condition_definitions": {
            condition["key"]: condition for condition in FINAL_CONDITIONS
        },
        "comparison": rows,
        "deltas_vs_nominal": deltas,
        "conditions": summaries,
    }
    (output_directory / "final_summary.json").write_text(
        json.dumps(payload, indent=2)
    )

    csv_fields = list(rows[0])
    with (output_directory / "final_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    table = format_final_comparison(rows)
    (output_directory / "final_comparison.txt").write_text(table + "\n")
    return payload, table
