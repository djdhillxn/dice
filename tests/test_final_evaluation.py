"""Unit tests for final-evaluation reuse and comparison artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from dicedial.final_evaluation import (
    FINAL_CONDITIONS,
    final_evaluation_is_reusable,
    sha256_file,
    write_final_evaluation_artifacts,
)


def _summary(checkpoint, condition, success, drop, mean_commands):
    attempts = 1_000
    successful = round(success * attempts)
    return {
        "project": "DICE",
        "task": condition["task"],
        "checkpoint": str(checkpoint),
        "checkpoint_source": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": condition["seed"],
        "num_envs": 256,
        "episodes": 1_000,
        "successful_commands": successful,
        "attempted_commands": attempts,
        "issued_command_completion_rate": success,
        "target_face_success_rate": success,
        "episode_any_completion_fraction": 0.95,
        "zero_completion_episode_fraction": 0.05,
        "completed_commands_per_sim_minute": 80.0,
        "median_time_to_target_seconds": 0.7,
        "drop_rate": drop,
        "mean_consecutive_commands": mean_commands,
        "median_consecutive_commands": mean_commands,
        "max_consecutive_commands": 40,
        "deterministic_action_out_of_bounds_fraction": 0.2,
        "per_face": {
            str(face): {
                "successes": 100,
                "attempts": 101,
                "success_rate": success - face / 10_000.0,
            }
            for face in range(1, 7)
        },
    }


class TestFinalEvaluation(unittest.TestCase):
    def test_reuse_requires_checkpoint_content_and_full_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model_4000.pt"
            checkpoint.write_bytes(b"selected checkpoint")
            condition = FINAL_CONDITIONS[0]
            summary_path = root / "nominal" / "summary.json"
            summary_path.parent.mkdir()
            summary = _summary(checkpoint, condition, 0.97, 0.09, 33.0)
            # Isaac Lab may transparently convert an older checkpoint before
            # loading it. Reuse identity must follow the original source.
            summary["checkpoint"] = str(root / "converted_model.pt")
            summary_path.write_text(json.dumps(summary))

            self.assertTrue(
                final_evaluation_is_reusable(
                    summary_path, checkpoint, "nominal", 1_000, 256
                )
            )
            self.assertFalse(
                final_evaluation_is_reusable(
                    summary_path, checkpoint, "nominal", 500, 256
                )
            )

            checkpoint.write_bytes(b"different checkpoint")
            self.assertFalse(
                final_evaluation_is_reusable(
                    summary_path, checkpoint, "nominal", 1_000, 256
                )
            )

    def test_writes_three_condition_comparison_and_deltas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model_4000.pt"
            checkpoint.write_bytes(b"selected checkpoint")
            values = {
                "nominal": (0.97, 0.09, 33.0),
                "robust": (0.96, 0.11, 30.0),
                "adverse": (0.80, 0.30, 12.0),
            }
            paths = {}
            for condition in FINAL_CONDITIONS:
                key = condition["key"]
                path = root / key / "summary.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps(_summary(checkpoint, condition, *values[key]))
                )
                paths[key] = path

            output = root / "combined"
            payload, table = write_final_evaluation_artifacts(paths, output)

            self.assertEqual(
                payload["condition_order"], ["nominal", "robust", "adverse"]
            )
            self.assertAlmostEqual(
                payload["deltas_vs_nominal"]["adverse"]["drop_rate_delta"],
                0.21,
            )
            self.assertIn("Adverse heavy/slippery", table)
            self.assertTrue((output / "final_summary.json").is_file())
            self.assertTrue((output / "final_comparison.csv").is_file())
            self.assertTrue((output / "final_comparison.txt").is_file())


if __name__ == "__main__":
    unittest.main()
