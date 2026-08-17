"""Pure helpers and contracts for the DICE portfolio-video pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


PORTFOLIO_SCHEMA_VERSION = 1
DEFAULT_RESOLUTION = (1920, 1080)
RAW_FPS = 60
EXPORT_FPS = 30
PORTFOLIO_PLAYBACK_SPEED = 0.5
PORTFOLIO_FINAL_HOLD_SECONDS = 0.75
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
# diagnostic views are intentionally fixed and static so repeated trajectory
# replays can be synchronized into full-height presentation panels.
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


def presentation_duration(
    simulation_duration,
    playback_speed=PORTFOLIO_PLAYBACK_SPEED,
    final_hold_seconds=PORTFOLIO_FINAL_HOLD_SECONDS,
):
    """Return the expected edited duration for one complete rollout."""

    simulation_duration = float(simulation_duration)
    playback_speed = float(playback_speed)
    final_hold_seconds = float(final_hold_seconds)
    if simulation_duration < 0.0:
        raise ValueError("Simulation duration must be non-negative")
    if playback_speed <= 0.0:
        raise ValueError("Playback speed must be positive")
    if final_hold_seconds < 0.0:
        raise ValueError("Final hold duration must be non-negative")
    return simulation_duration / playback_speed + final_hold_seconds


def resolve_portfolio_artifact(path, output_directory):
    """Resolve a manifest artifact after a portfolio directory has been copied."""

    output_directory = Path(output_directory).expanduser().resolve()
    recorded = Path(path).expanduser()
    if recorded.is_file():
        return recorded.resolve()

    matching_indices = [
        index
        for index, part in enumerate(recorded.parts)
        if part == output_directory.name
    ]
    if not matching_indices:
        raise FileNotFoundError(
            f"Portfolio artifact does not exist and cannot be rebased: {recorded}"
        )
    relative_parts = recorded.parts[matching_indices[-1] + 1 :]
    candidate = output_directory.joinpath(*relative_parts).resolve()
    try:
        candidate.relative_to(output_directory)
    except ValueError as exc:
        raise ValueError(
            f"Rebased artifact escapes the portfolio root: {recorded}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Rebased portfolio artifact does not exist: {candidate}"
        )
    return candidate


def build_portfolio_captions(final_results, selections, playback_speed=0.5):
    """Build the three copy-ready Markdown companions for public videos."""

    missing_results = {"nominal", "robust", "adverse"} - final_results.keys()
    missing_selections = {"nominal", "robust", "adverse"} - selections.keys()
    if missing_results or missing_selections:
        raise ValueError(
            "Caption inputs are incomplete: "
            f"results={sorted(missing_results)}, selections={sorted(missing_selections)}"
        )
    if playback_speed <= 0.0:
        raise ValueError("Playback speed must be positive")

    def percent(condition, field):
        return 100.0 * float(final_results[condition][field])

    def episodes(condition):
        return int(float(final_results[condition]["episodes"]))

    nominal = selections["nominal"]
    robust = selections["robust"]
    adverse = selections["adverse"]
    disclosure = (
        f"Footage is presented at **{playback_speed:g}× playback** for visual "
        "clarity; the deterministic policy was executed and evaluated at the "
        "original 60 Hz control rate."
    )

    nominal_caption = f"""# Nominal semantic success

The requested die face is supplied as a semantic command to one deterministic
20-DoF Shadow Hand policy. The synchronized oblique and top-down views show the
same action trajectory, allowing the manipulation and upward-facing result to
be checked together.

- **Representative rollout:** seed {int(nominal["seed"])}, selected as the successful six-command candidate nearest the median completion time
- **Rollout result:** {int(nominal["commands_completed"])} commands in {float(nominal["duration_seconds"]):.2f} simulation seconds, with no drop
- **Final evaluation:** {percent("nominal", "issued_command_completion_rate"):.2f}% issued-command completion, {percent("nominal", "drop_rate"):.2f}% episode drop rate, {float(final_results["nominal"]["mean_consecutive_commands"]):.3f} mean consecutive commands over {episodes("nominal"):,} episodes
- **Views:** oblique manipulation view (left) and top-down verification view (right)

{disclosure}

## What to watch

The target-face indicator changes only after the requested face is aligned,
stabilized, and held through all 20 confirmation steps. The top-down view makes
that semantic success independently visible.
"""

    variation_caption = f"""# Symmetric physics variation

This comparison places a representative nominal rollout beside a representative
held-out physics rollout. The variation samples die mass and static/dynamic
friction within ±20% of nominal; both use the same trained policy and the same
six-command semantic cycle.

- **Nominal rollout:** seed {int(nominal["seed"])}, {int(nominal["commands_completed"])} commands in {float(nominal["duration_seconds"]):.2f} simulation seconds
- **Variation rollout:** seed {int(robust["seed"])}, {int(robust["commands_completed"])} commands in {float(robust["duration_seconds"]):.2f} simulation seconds
- **Nominal evaluation:** {percent("nominal", "issued_command_completion_rate"):.2f}% command completion and {percent("nominal", "drop_rate"):.2f}% drops over {episodes("nominal"):,} episodes
- **Variation evaluation:** {percent("robust", "issued_command_completion_rate"):.2f}% command completion and {percent("robust", "drop_rate"):.2f}% drops over {episodes("robust"):,} episodes

{disclosure}

## Interpretation

The selected clips are median-like examples rather than the fastest trials.
The aggregate evaluation—not either individual video—is the evidence for
generalization under the held-out variation.
"""

    adverse_caption = f"""# Adverse retention boundary

The adverse condition fixes die mass at 1.5× nominal and both object-friction
coefficients at 0.7× nominal. Synchronized oblique and side-contact views show a
representative rollout that completes multiple semantic commands before the
grasp eventually fails.

- **Representative rollout:** seed {int(adverse["seed"])}, selected to match the median failed episode in the final evaluation
- **Rollout result:** {int(adverse["commands_completed"])} commands before dropping at {float(adverse["duration_seconds"]):.2f} simulation seconds
- **Final evaluation:** {percent("adverse", "issued_command_completion_rate"):.2f}% issued-command completion, {percent("adverse", "drop_rate"):.2f}% episode drop rate, {float(final_results["adverse"]["mean_consecutive_commands"]):.3f} mean consecutive commands over {episodes("adverse"):,} episodes
- **Views:** oblique manipulation view (left) and side contact/failure view (right)

{disclosure}

## Interpretation

Semantic targeting remains strong, but long-horizon grasp retention degrades
at this deliberately difficult heavy-and-slippery corner. This negative result
identifies a concrete sim-to-real robustness direction rather than hiding the
policy's failure boundary.
"""

    return {
        "dice_nominal_success.md": nominal_caption,
        "dice_physics_variation.md": variation_caption,
        "dice_adverse_boundary.md": adverse_caption,
    }


def probe_portfolio_video(ffprobe, path, expected_resolution, expected_fps):
    """Validate one browser-ready MP4 and return its publication metadata."""

    path = Path(path)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    video_streams = [
        item for item in payload["streams"] if item["codec_type"] == "video"
    ]
    audio_streams = [
        item for item in payload["streams"] if item["codec_type"] == "audio"
    ]
    if len(video_streams) != 1 or audio_streams:
        raise ValueError(f"Expected one video and no audio streams: {path}")

    video_stream = video_streams[0]
    if video_stream["codec_name"] != "h264":
        raise ValueError(f"Expected H.264, got {video_stream['codec_name']}: {path}")
    if video_stream.get("pix_fmt") != "yuv420p":
        raise ValueError(f"Expected yuv420p, got {video_stream.get('pix_fmt')}: {path}")
    resolution = (int(video_stream["width"]), int(video_stream["height"]))
    if resolution != tuple(expected_resolution):
        raise ValueError(f"Unexpected resolution for {path}")
    numerator, denominator = video_stream["avg_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=0.01):
        raise ValueError(f"Expected {expected_fps} FPS, got {fps}: {path}")

    size_bytes = path.stat().st_size
    if size_bytes > 50 * 1024 * 1024:
        raise ValueError(f"Portfolio export exceeds 50 MiB: {path}")
    with path.open("rb") as mp4_file:
        atom_prefix = mp4_file.read(min(size_bytes, 4 * 1024 * 1024))
    moov_offset = atom_prefix.find(b"moov")
    mdat_offset = atom_prefix.find(b"mdat")
    faststart = moov_offset >= 0 and mdat_offset >= 0 and moov_offset < mdat_offset
    if not faststart:
        raise ValueError(f"MP4 metadata is not fast-start optimized: {path}")

    return {
        "path": str(path),
        "codec": video_stream["codec_name"],
        "pixel_format": video_stream.get("pix_fmt"),
        "resolution": list(resolution),
        "fps": fps,
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": size_bytes,
        "sha256": sha256_file(path),
        "faststart": faststart,
    }


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


def align_video_telemetry(
    metrics,
    initial_metrics,
    decodable_frame_count,
    frame_origin="legacy_auto",
):
    """Align capture telemetry to decoded frames without hiding real drift.

    Gymnasium's MoviePy-backed recorder can omit the final encoded frame even
    though the terminal environment step was recorded in telemetry.  That
    boundary case is safe to repair by drawing the terminal HUD over a copy of
    the last decodable frame.  Any larger discrepancy, or a missing
    non-terminal row, remains a hard failure.
    """

    metric_count = len(metrics)
    terminal_row = metrics[-1] if metric_count else None
    try:
        final_row_is_terminal = (
            terminal_row is not None and int(float(terminal_row.get("done", 0))) == 1
        )
    except (AttributeError, TypeError, ValueError):
        final_row_is_terminal = False

    if (
        frame_origin == "initial_plus_post_step"
        and decodable_frame_count == metric_count + 1
    ):
        initial = {key: str(value) for key, value in initial_metrics.items()}
        return {
            "frame_rows": [initial, *metrics],
            "synthesized_rows": [],
            "mode": "initial_plus_post_step",
        }
    if (
        frame_origin == "initial_plus_post_step"
        and decodable_frame_count == metric_count
        and final_row_is_terminal
    ):
        initial = {key: str(value) for key, value in initial_metrics.items()}
        return {
            "frame_rows": [initial, *metrics[:-1]],
            "synthesized_rows": [terminal_row],
            "mode": "initial_plus_post_step_plus_synthesized_terminal",
        }
    if (
        decodable_frame_count == metric_count
        and frame_origin != "initial_plus_post_step"
    ):
        return {
            "frame_rows": list(metrics),
            "synthesized_rows": [],
            "mode": "post_step",
        }
    if decodable_frame_count == metric_count + 1 and frame_origin == "post_step":
        if metric_count == 0:
            raise RuntimeError("Cannot align a post-step video to empty telemetry")
        return {
            "frame_rows": [*metrics, metrics[-1]],
            "synthesized_rows": [],
            "mode": "post_step_plus_encoded_terminal_duplicate",
        }
    if decodable_frame_count == metric_count + 1 and frame_origin == "legacy_auto":
        initial = {key: str(value) for key, value in initial_metrics.items()}
        return {
            "frame_rows": [initial, *metrics],
            "synthesized_rows": [],
            "mode": "initial_plus_post_step",
        }
    if (
        decodable_frame_count == metric_count - 1
        and frame_origin != "initial_plus_post_step"
        and final_row_is_terminal
    ):
        return {
            "frame_rows": list(metrics[:-1]),
            "synthesized_rows": [terminal_row],
            "mode": "post_step_plus_synthesized_terminal",
        }

    raise RuntimeError(
        "Video/telemetry synchronization failed: "
        f"video has {decodable_frame_count} decodable frames, metrics has "
        f"{metric_count} rows (declared origin: {frame_origin}). Supported "
        "mappings are exact post-step, one "
        "explicit initial frame (when declared), one encoded terminal duplicate "
        "for post-step capture, or one missing final frame when the unmatched "
        "telemetry row is terminal."
    )


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
