"""Unit tests for checkpoint discovery and selection ranking."""

import json
import tempfile
import unittest
from pathlib import Path

from dicedial.checkpoint_sweep import (
    discover_checkpoints,
    evaluation_is_reusable,
    rank_entries,
    resolve_run_directory,
    write_ranking_artifacts,
)


def _entry(checkpoint, success, drop, commands, minimum_face, latency, action_oob):
    return {
        "checkpoint": checkpoint,
        "summary": f"{checkpoint}/summary.json",
        "issued_command_completion_rate": success,
        "drop_rate": drop,
        "mean_consecutive_commands": commands,
        "minimum_per_face_success_rate": minimum_face,
        "minimum_per_face": "1",
        "median_time_to_target_seconds": latency,
        "deterministic_action_out_of_bounds_fraction": action_oob,
        "per_face_success_rate": {str(face): minimum_face for face in range(1, 7)},
    }


class TestCheckpointSweep(unittest.TestCase):
    def test_resolve_run_id_and_discover_training_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = Path(temporary_directory) / "outputs"
            run_directory = outputs / "2026-08-16_12-00-00_example"
            run_directory.mkdir(parents=True)
            for name in (
                "model_0.pt",
                "model_2000.pt",
                "model_1000.pt",
                "model_final.pt",
                "model_invalid.pt",
            ):
                (run_directory / name).touch()

            resolved = resolve_run_directory(run_directory.name, outputs)
            checkpoints = discover_checkpoints(resolved)

            self.assertEqual(resolved, run_directory.resolve())
            self.assertEqual(
                [checkpoint.name for checkpoint in checkpoints],
                ["model_1000.pt", "model_2000.pt", "model_final.pt"],
            )

    def test_rank_prioritizes_threshold_then_documented_tiebreakers(self):
        entries = [
            _entry("below.pt", 0.89, 0.01, 20.0, 0.88, 0.5, 0.01),
            _entry("riskier.pt", 0.95, 0.20, 10.0, 0.90, 0.8, 0.10),
            _entry("winner.pt", 0.91, 0.05, 6.0, 0.87, 1.0, 0.15),
            _entry("fewer_commands.pt", 0.94, 0.05, 5.0, 0.99, 0.3, 0.01),
        ]

        ranked = rank_entries(entries, success_threshold=0.90)

        self.assertEqual(
            [entry["checkpoint"] for entry in ranked],
            ["winner.pt", "fewer_commands.pt", "riskier.pt", "below.pt"],
        )
        self.assertTrue(ranked[0]["meets_command_success_threshold"])
        self.assertFalse(ranked[-1]["meets_command_success_threshold"])

    def test_reuse_validation_and_ranking_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model_1000.pt"
            checkpoint.touch()
            summary_path = root / "model_1000" / "nominal" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary = {
                "checkpoint": str(checkpoint),
                "task": "DICE-Shadow-Eval-v0",
                "seed": 2026,
                "num_envs": 256,
                "episodes": 500,
                "issued_command_completion_rate": 0.91,
                "drop_rate": 0.04,
                "mean_consecutive_commands": 4.5,
                "median_time_to_target_seconds": 1.0,
                "deterministic_action_out_of_bounds_fraction": 0.08,
                "per_face": {
                    str(face): {"success_rate": 0.90 + face / 1000.0}
                    for face in range(1, 7)
                },
            }
            summary_path.write_text(json.dumps(summary))

            self.assertTrue(
                evaluation_is_reusable(
                    summary_path,
                    checkpoint,
                    "DICE-Shadow-Eval-v0",
                    2026,
                    500,
                    256,
                )
            )
            self.assertFalse(
                evaluation_is_reusable(
                    summary_path,
                    checkpoint,
                    "DICE-Shadow-Eval-v0",
                    2027,
                    500,
                    256,
                )
            )

            output_directory = root / "ranking"
            payload, table = write_ranking_artifacts([summary_path], output_directory)

            self.assertEqual(payload["selected_checkpoint"], "model_1000.pt")
            self.assertIn("model_1000.pt", table)
            self.assertTrue((output_directory / "ranking.json").is_file())
            self.assertTrue((output_directory / "ranking.csv").is_file())
            self.assertTrue((output_directory / "ranking.txt").is_file())
            self.assertEqual(
                (output_directory / "selected_checkpoint.txt").read_text(),
                "model_1000.pt\n",
            )


if __name__ == "__main__":
    unittest.main()
