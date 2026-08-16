"""Run the final nominal, symmetric-robust, and adverse DICE evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dicedial.final_evaluation import (
    FINAL_CONDITIONS,
    final_evaluation_is_reusable,
    write_final_evaluation_artifacts,
)
from dicedial.training_artifacts import write_artifact_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete three-condition DICE final evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="Selected RSL-RL model_*.pt checkpoint")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to <checkpoint-run>/evaluation/final_<checkpoint-stem>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun matching completed conditions instead of reusing them.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    return args


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    run_directory = checkpoint.parent
    output_directory = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_directory / "evaluation" / f"final_{checkpoint.stem}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    state_path = output_directory / "evaluation_run.json"
    state = {
        "schema_version": 1,
        "project": "DICE",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint),
        "episodes_per_condition": args.episodes,
        "num_envs": args.num_envs,
        "device": args.device,
        "conditions": [condition["key"] for condition in FINAL_CONDITIONS],
        "completed": [],
        "reused": [],
    }
    state_path.write_text(json.dumps(state, indent=2))

    print(f"[DICE FINAL] Checkpoint: {checkpoint}", flush=True)
    print(f"[DICE FINAL] Output: {output_directory}", flush=True)
    print(
        f"[DICE FINAL] {args.episodes} episodes per condition, "
        f"{args.num_envs} parallel envs",
        flush=True,
    )

    summary_paths = {}
    try:
        for index, condition in enumerate(FINAL_CONDITIONS, start=1):
            key = condition["key"]
            condition_directory = output_directory / key
            summary_path = condition_directory / "summary.json"
            reusable = final_evaluation_is_reusable(
                summary_path,
                checkpoint,
                key,
                args.episodes,
                args.num_envs,
            )
            print(
                f"\n[DICE FINAL] [{index}/{len(FINAL_CONDITIONS)}] "
                f"{condition['label']} ({condition['task']})",
                flush=True,
            )
            if reusable and not args.force:
                print(f"[DICE FINAL] Reusing: {summary_path}", flush=True)
                state["reused"].append(key)
            else:
                state["active_condition"] = key
                state_path.write_text(json.dumps(state, indent=2))
                command = [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts" / "evaluate_rsl.py"),
                    "--task",
                    condition["task"],
                    "--checkpoint",
                    str(checkpoint),
                    "--episodes",
                    str(args.episodes),
                    "--num_envs",
                    str(args.num_envs),
                    "--seed",
                    str(condition["seed"]),
                    "--device",
                    args.device,
                    "--headless",
                    "--output",
                    str(condition_directory),
                ]
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                if not final_evaluation_is_reusable(
                    summary_path,
                    checkpoint,
                    key,
                    args.episodes,
                    args.num_envs,
                ):
                    raise RuntimeError(
                        "Evaluation summary does not match the requested "
                        f"checkpoint/condition contract: {summary_path}"
                    )
                state["completed"].append(key)
                state["active_condition"] = None
            if not summary_path.is_file():
                raise RuntimeError(
                    f"Evaluation did not produce expected summary: {summary_path}"
                )
            summary_paths[key] = summary_path
            state_path.write_text(json.dumps(state, indent=2))

        payload, table = write_final_evaluation_artifacts(
            summary_paths, output_directory
        )
        state.update(
            {
                "status": "complete",
                "finished_at": datetime.now().astimezone().isoformat(),
                "active_condition": None,
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
        state_path.write_text(json.dumps(state, indent=2))
        write_artifact_manifest(run_directory)

        print("\n[DICE FINAL] Final comparison", flush=True)
        print(table, flush=True)
        print(f"\n[DICE FINAL] Results: {output_directory}", flush=True)
    except KeyboardInterrupt:
        state.update(
            {
                "status": "interrupted",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        state_path.write_text(json.dumps(state, indent=2))
        write_artifact_manifest(run_directory)
        raise
    except Exception:
        state.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
            }
        )
        state_path.write_text(json.dumps(state, indent=2))
        write_artifact_manifest(run_directory)
        raise


if __name__ == "__main__":
    main()
