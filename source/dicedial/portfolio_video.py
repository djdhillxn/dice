"""Pure helpers and contracts for the DICE portfolio-video pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PORTFOLIO_SCHEMA_VERSION = 1
DEFAULT_RESOLUTION = (1920, 1080)
RAW_FPS = 60
EXPORT_FPS = 30
PRESENTATION_COLLISION_EXTENT_M = (0.060, 0.060, 0.060)
PRESENTATION_MASS_KG = 0.216

PORTFOLIO_CONDITIONS = {
    "nominal": {
        "key": "nominal",
        "label": "NOMINAL",
        "task": "DICE-Shadow-Play-v0",
        "description": "Numbered presentation die at nominal physics.",
        "physics": {
            "object_mass_scale": 1.0,
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
        },
        "max_steps": 2_400,
    },
    "robust": {
        "key": "robust",
        "label": "SYMMETRIC PHYSICS VARIATION",
        "task": "DICE-Shadow-Play-Robust-v0",
        "description": "Numbered die with held-out mass/material samples.",
        "physics": {
            "object_mass_scale": {"distribution": "uniform", "range": [0.8, 1.2]},
            "static_friction": {"distribution": "uniform", "range": [0.8, 1.2]},
            "dynamic_friction": {
                "distribution": "uniform",
                "range": [0.8, 1.2],
                "constraint": "dynamic <= static",
            },
        },
        "max_steps": 2_400,
    },
    "adverse": {
        "key": "adverse",
        "label": "ADVERSE: HEAVY / LOW FRICTION",
        "task": "DICE-Shadow-Play-Adverse-v0",
        "description": "Numbered die at the fixed heavy/slippery stress corner.",
        "physics": {
            "object_mass_scale": 1.5,
            "static_friction": 0.7,
            "dynamic_friction": 0.7,
        },
        "max_steps": 1_440,
    },
}

# The hero preset preserves the existing known-good presentation view. The two
# diagnostic views are intentionally fixed and static; a contact sheet is still
# produced so their framing can be reviewed before final rendering.
CAMERA_PRESETS = {
    "hero": {
        "label": "Hero oblique",
        "eye": (0.95, -1.20, 0.88),
        "lookat": (0.0, -0.39, 0.59),
    },
    "top": {
        "label": "Top diagnostic",
        "eye": (0.10, -0.54, 1.48),
        "lookat": (0.0, -0.39, 0.59),
    },
    "side": {
        "label": "Side contact",
        "eye": (1.12, -0.31, 0.70),
        "lookat": (0.0, -0.39, 0.59),
    },
}

TRACE_EVENT_FIELDS = (
    "target_face",
    "top_face",
    "success",
    "completed_face",
    "commands_completed",
    "drop",
    "done",
)
TRACE_NUMERIC_FIELDS = (
    "alignment",
    "position_error",
    "hold_progress",
)


def sha256_file(path):
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_resolution(value):
    """Parse ``WIDTHxHEIGHT`` and return a positive integer tuple."""

    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Resolution must use WIDTHxHEIGHT, for example 1920x1080"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError("Resolution dimensions must be positive")
    if width % 2 or height % 2:
        raise ValueError("Resolution dimensions must be even for yuv420p encoding")
    return width, height


def parse_seed_spec(value):
    """Parse comma-separated seeds and inclusive ``start:end`` ranges."""

    seeds = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            start_text, end_text = token.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    deduplicated = list(dict.fromkeys(seeds))
    if not deduplicated:
        raise ValueError("At least one seed is required")
    if any(seed < 0 for seed in deduplicated):
        raise ValueError("Seeds must be non-negative")
    return deduplicated


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _median(values):
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def select_representative_scout(
    condition,
    summaries,
    adverse_target_commands=9.0,
    adverse_target_duration_seconds=7.28,
):
    """Select a deterministic representative using the documented rule."""

    if condition not in PORTFOLIO_CONDITIONS:
        raise KeyError(f"Unknown portfolio condition: {condition}")
    complete = [summary for summary in summaries if summary.get("status") == "complete"]
    if condition == "adverse":
        candidates = [summary for summary in complete if summary.get("dropped")]
        if not candidates:
            raise ValueError("No dropped adverse scout was found")

        # Match the failed-episode medians from the quantitative evaluation.
        # Command distance is the primary criterion.
        return min(
            candidates,
            key=lambda summary: (
                abs(
                    float(summary["commands_completed"])
                    - float(adverse_target_commands)
                ),
                abs(
                    float(summary["duration_seconds"])
                    - float(adverse_target_duration_seconds)
                ),
                int(summary["seed"]),
            ),
        )

    candidates = [
        summary
        for summary in complete
        if summary.get("outcome") == "completed_sequence"
        and int(summary.get("commands_completed", 0)) >= 6
    ]
    if not candidates:
        raise ValueError(f"No successful {condition} six-command scout was found")
    median_duration = _median(
        [float(summary["duration_seconds"]) for summary in candidates]
    )
    return min(
        candidates,
        key=lambda summary: (
            abs(float(summary["duration_seconds"]) - median_duration),
            int(summary["seed"]),
        ),
    )


def read_metric_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_metric_traces(reference_path, candidate_path, numeric_tolerance=1.0e-4):
    """Validate that camera replay retained the selected simulator trajectory."""

    reference = read_metric_rows(reference_path)
    candidate = read_metric_rows(candidate_path)
    if len(reference) != len(candidate):
        raise ValueError(
            f"Trace length mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )

    max_numeric_error = {field: 0.0 for field in TRACE_NUMERIC_FIELDS}
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        for field in TRACE_EVENT_FIELDS:
            if expected[field] != actual[field]:
                raise ValueError(
                    f"Trace event mismatch at row {index}, {field}: "
                    f"{expected[field]} != {actual[field]}"
                )
        for field in TRACE_NUMERIC_FIELDS:
            error = abs(float(expected[field]) - float(actual[field]))
            max_numeric_error[field] = max(max_numeric_error[field], error)
            if not math.isfinite(error) or error > numeric_tolerance:
                raise ValueError(
                    f"Trace numeric mismatch at row {index}, {field}: "
                    f"error={error:.6g} > {numeric_tolerance:.6g}"
                )
    return {
        "rows": len(reference),
        "numeric_tolerance": numeric_tolerance,
        "max_numeric_error": max_numeric_error,
    }


def compare_physics_snapshots(
    reference,
    presentation,
    relative_tolerance=0.02,
    absolute_tolerance=1.0e-8,
):
    """Require equivalent stock and numbered-die mass/inertia snapshots."""

    comparisons = {}
    failures = []
    for field in ("mass", "inertia"):
        expected = reference.get(field)
        actual = presentation.get(field)
        if expected is None or actual is None:
            failures.append(
                f"Physics {field} is unavailable; the presentation audit "
                "cannot establish equivalence"
            )
            comparisons[field] = {"status": "unavailable"}
            continue
        expected_values = [float(value) for value in expected]
        actual_values = [float(value) for value in actual]
        if len(expected_values) != len(actual_values):
            failures.append(
                f"Physics {field} shape mismatch: stock={len(expected_values)}, "
                f"numbered={len(actual_values)}"
            )
            comparisons[field] = {"status": "shape_mismatch"}
            continue
        errors = []
        violations = []
        for expected_value, actual_value in zip(expected_values, actual_values):
            absolute_error = abs(actual_value - expected_value)
            denominator = max(
                abs(expected_value), absolute_tolerance / relative_tolerance
            )
            errors.append(absolute_error / denominator)
            violations.append(
                absolute_error
                > max(
                    absolute_tolerance,
                    relative_tolerance * abs(expected_value),
                )
            )
        maximum = max(errors, default=0.0)
        maximum_index = errors.index(maximum) if errors else 0
        comparisons[field] = {
            "status": "mismatch" if any(violations) else "match",
            "maximum_relative_error": maximum,
        }
        if any(violations):
            failures.append(
                f"Presentation die {field} differs from stock by {100.0 * maximum:.2f}% "
                f"(relative limit {100.0 * relative_tolerance:.2f}%, "
                f"absolute floor {absolute_tolerance:.3g}; "
                f"stock={expected_values[maximum_index]:.9g}, "
                f"numbered={actual_values[maximum_index]:.9g})"
            )
    if failures:
        raise ValueError("Physics audit failed:\n- " + "\n- ".join(failures))
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "comparisons": comparisons,
    }
