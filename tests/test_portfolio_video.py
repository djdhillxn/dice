"""Unit tests for deterministic portfolio selection and validation helpers."""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dicedial.portfolio_video import (
    PRESENTATION_COLLISION_EXTENT_M,
    PRESENTATION_MASS_KG,
    align_video_telemetry,
    compare_metric_traces,
    compare_physics_snapshots,
    parse_resolution,
    parse_seed_spec,
    probe_portfolio_video,
    select_representative_scout,
)


def _scout(seed, duration, commands, outcome, dropped=False):
    return {
        "status": "complete",
        "seed": seed,
        "duration_seconds": duration,
        "commands_completed": commands,
        "outcome": outcome,
        "dropped": dropped,
    }


def _write_trace(path, alignment_offset=0.0, event_override=None):
    fields = (
        "target_face",
        "top_face",
        "success",
        "completed_face",
        "commands_completed",
        "drop",
        "done",
        "alignment",
        "position_error",
        "hold_progress",
    )
    rows = [
        {
            "target_face": 1,
            "top_face": 6,
            "success": 0,
            "completed_face": 0,
            "commands_completed": 0,
            "drop": 0,
            "done": 0,
            "alignment": 0.8 + alignment_offset,
            "position_error": 0.02,
            "hold_progress": 0.0,
        },
        {
            "target_face": 6,
            "top_face": 1,
            "success": 1,
            "completed_face": 1,
            "commands_completed": 1,
            "drop": 0,
            "done": 0,
            "alignment": 0.99 + alignment_offset,
            "position_error": 0.02,
            "hold_progress": 1.0,
        },
    ]
    if event_override:
        rows[1].update(event_override)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestPortfolioVideo(unittest.TestCase):
    def test_probe_returns_video_metadata_after_faststart_file_check(self):
        payload = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30/1",
                }
            ],
            "format": {"duration": "12.5"},
        }
        completed = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            video = Path(temporary_directory) / "portfolio.mp4"
            video.write_bytes(b"ftyp" + b"moov" + b"metadata" + b"mdat" + b"video")
            with patch(
                "dicedial.portfolio_video.subprocess.run",
                return_value=completed,
            ) as run:
                metadata = probe_portfolio_video(
                    "ffprobe",
                    video,
                    (1920, 1080),
                    30,
                )

        self.assertEqual(metadata["codec"], "h264")
        self.assertEqual(metadata["pixel_format"], "yuv420p")
        self.assertEqual(metadata["resolution"], [1920, 1080])
        self.assertEqual(metadata["fps"], 30.0)
        self.assertEqual(metadata["duration_seconds"], 12.5)
        self.assertTrue(metadata["faststart"])
        run.assert_called_once()

    def test_video_telemetry_alignment_accepts_only_safe_boundary_cases(self):
        metrics = [
            {"step": "0", "done": "0"},
            {"step": "1", "done": "0"},
            {"step": "2", "done": "1", "drop": "1"},
        ]
        initial = {"step": -1, "done": 0}

        exact = align_video_telemetry(metrics, initial, 3)
        self.assertEqual(exact["mode"], "post_step")
        self.assertEqual(exact["frame_rows"], metrics)
        self.assertEqual(exact["synthesized_rows"], [])

        with_initial = align_video_telemetry(
            metrics,
            initial,
            4,
            frame_origin="initial_plus_post_step",
        )
        self.assertEqual(with_initial["mode"], "initial_plus_post_step")
        self.assertEqual(with_initial["frame_rows"][0]["step"], "-1")

        initial_missing_terminal = align_video_telemetry(
            metrics,
            initial,
            3,
            frame_origin="initial_plus_post_step",
        )
        self.assertEqual(
            initial_missing_terminal["mode"],
            "initial_plus_post_step_plus_synthesized_terminal",
        )
        self.assertEqual(initial_missing_terminal["frame_rows"][0]["step"], "-1")
        self.assertEqual(initial_missing_terminal["synthesized_rows"], metrics[-1:])

        duplicated_terminal = align_video_telemetry(
            metrics,
            initial,
            4,
            frame_origin="post_step",
        )
        self.assertEqual(
            duplicated_terminal["mode"],
            "post_step_plus_encoded_terminal_duplicate",
        )
        self.assertEqual(duplicated_terminal["frame_rows"][-1], metrics[-1])

        missing_terminal = align_video_telemetry(
            metrics,
            initial,
            2,
            frame_origin="post_step",
        )
        self.assertEqual(
            missing_terminal["mode"], "post_step_plus_synthesized_terminal"
        )
        self.assertEqual(missing_terminal["frame_rows"], metrics[:-1])
        self.assertEqual(missing_terminal["synthesized_rows"], metrics[-1:])

        with self.assertRaises(RuntimeError):
            align_video_telemetry(metrics[:-1], initial, 1)
        with self.assertRaises(RuntimeError):
            align_video_telemetry(metrics, initial, 1)

    def test_numbered_die_collision_matches_stock_dexcube_extent(self):
        asset = (
            Path(__file__).parents[1]
            / "source"
            / "dicedial"
            / "assets"
            / "numbered_die.usda"
        ).read_text(encoding="utf-8")
        self.assertEqual(PRESENTATION_COLLISION_EXTENT_M, (0.060, 0.060, 0.060))
        self.assertEqual(PRESENTATION_MASS_KG, 0.216)
        self.assertIn('"PhysicsMassAPI"', asset)
        self.assertIn("float physics:mass = 0.216", asset)
        self.assertNotIn("physics:density", asset)
        self.assertIn("double size = 0.060", asset)
        self.assertNotIn("double size = 0.065", asset)
        self.assertIn("double size = 0.059", asset)
        self.assertNotIn("0.0325", asset)

    def test_parses_resolution_and_seed_ranges(self):
        self.assertEqual(parse_resolution("1920x1080"), (1920, 1080))
        self.assertEqual(parse_seed_spec("7:9,11,9"), [7, 8, 9, 11])
        with self.assertRaises(ValueError):
            parse_resolution("1919x1080")
        with self.assertRaises(ValueError):
            parse_seed_spec("9:7")

    def test_selects_median_success_and_representative_adverse_drop(self):
        nominal = [
            _scout(7, 5.0, 6, "completed_sequence"),
            _scout(8, 7.0, 6, "completed_sequence"),
            _scout(9, 6.0, 6, "completed_sequence"),
            _scout(10, 2.0, 1, "dropped", True),
        ]
        selected_nominal = select_representative_scout("nominal", nominal)
        self.assertEqual(selected_nominal["seed"], 9)

        adverse = [
            _scout(20, 2.0, 0, "dropped", True),
            _scout(21, 7.4, 9, "dropped", True),
            _scout(22, 9.0, 10, "dropped", True),
            _scout(23, 24.0, 32, "timeout"),
        ]
        selected_adverse = select_representative_scout("adverse", adverse)
        self.assertEqual(selected_adverse["seed"], 21)
        selected_custom_adverse = select_representative_scout(
            "adverse",
            adverse,
            adverse_target_commands=10,
            adverse_target_duration_seconds=9.1,
        )
        self.assertEqual(selected_custom_adverse["seed"], 22)

    def test_trace_validation_requires_events_and_numeric_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.csv"
            matching = root / "matching.csv"
            divergent_number = root / "divergent_number.csv"
            divergent_event = root / "divergent_event.csv"
            _write_trace(reference)
            _write_trace(matching, alignment_offset=1.0e-5)
            _write_trace(divergent_number, alignment_offset=1.0e-2)
            _write_trace(divergent_event, event_override={"target_face": 2})

            result = compare_metric_traces(reference, matching)
            self.assertEqual(result["rows"], 2)
            with self.assertRaises(ValueError):
                compare_metric_traces(reference, divergent_number)
            with self.assertRaises(ValueError):
                compare_metric_traces(reference, divergent_event)

    def test_physics_snapshot_comparison(self):
        reference = {"mass": [0.15], "inertia": [0.1, 0.2, 0.3]}
        matching = {"mass": [0.151], "inertia": [0.1, 0.202, 0.3]}
        result = compare_physics_snapshots(reference, matching)
        self.assertEqual(result["comparisons"]["mass"]["status"], "match")
        with self.assertRaises(ValueError):
            compare_physics_snapshots(
                reference,
                {"mass": [0.18], "inertia": [0.1, 0.2, 0.3]},
            )
        with self.assertRaises(ValueError):
            compare_physics_snapshots(
                reference,
                {"mass": None, "inertia": [0.1, 0.2, 0.3]},
            )

        near_zero_reference = {"mass": [0.15], "inertia": [0.0, 0.0, 1.0e-4]}
        near_zero_matching = {
            "mass": [0.15],
            "inertia": [1.0e-9, 0.0, 1.01e-4],
        }
        result = compare_physics_snapshots(
            near_zero_reference,
            near_zero_matching,
        )
        self.assertEqual(result["comparisons"]["inertia"]["status"], "match")

        with self.assertRaisesRegex(ValueError, "(?s)mass.*inertia"):
            compare_physics_snapshots(
                reference,
                {"mass": [0.18], "inertia": [0.2, 0.3, 0.4]},
            )


if __name__ == "__main__":
    unittest.main()
