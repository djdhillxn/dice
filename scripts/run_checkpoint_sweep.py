"""Evaluate and rank every saved checkpoint from one completed DICE run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dicedial.checkpoint_sweep import (
    discover_checkpoints,
    evaluation_is_reusable,
    resolve_run_directory,
    write_ranking_artifacts,
)
from dicedial.training_artifacts import write_artifact_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run nominal frozen-policy evaluation for every saved checkpoint and "
            "rank the results without overwriting per-checkpoint artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run",
        help=(
            "Run ID below outputs/ (for example 2026-08-16_12-00-00_alias) "
            "or a direct run-directory path."
        ),
    )
    parser.add_argument("--outputs-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--success-threshold", type=float, default=0.90)
    parser.add_argument(
        "--include-initial",
        action="store_true",
        help="Also evaluate model_0.pt, which is excluded by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun matching completed evaluations instead of reusing them.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if not 0.0 <= args.success_threshold <= 1.0:
        parser.error("--success-threshold must be between 0 and 1")
    return args


def main():
    args = parse_args()
    run_directory = resolve_run_directory(args.run, args.outputs_root)
    checkpoints = discover_checkpoints(
        run_directory, include_initial=args.include_initial
    )
    sweep_directory = run_directory / "evaluation" / "checkpoint_sweep"
    sweep_directory.mkdir(parents=True, exist_ok=True)

    print(f"[DICE SWEEP] Run directory: {run_directory}", flush=True)
    print(
        "[DICE SWEEP] Checkpoints: "
        + ", ".join(checkpoint.name for checkpoint in checkpoints),
        flush=True,
    )
    print(
        f"[DICE SWEEP] Evaluating sequentially: {args.episodes} episodes, "
        f"{args.num_envs} envs, seed {args.seed}",
        flush=True,
    )

    state_path = sweep_directory / "sweep.json"
    state = {
        "project": "DICE",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "run_directory": str(run_directory),
        "task": "DICE-Shadow-Eval-v0",
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "success_threshold": args.success_threshold,
        "include_initial": args.include_initial,
        "checkpoints": [checkpoint.name for checkpoint in checkpoints],
        "completed": [],
        "reused": [],
    }
    state_path.write_text(json.dumps(state, indent=2))

    summary_paths = []
    try:
        for index, checkpoint in enumerate(checkpoints, start=1):
            output_directory = sweep_directory / checkpoint.stem / "nominal"
            summary_path = output_directory / "summary.json"
            reusable = evaluation_is_reusable(
                summary_path,
                checkpoint,
                "DICE-Shadow-Eval-v0",
                args.seed,
                args.episodes,
                args.num_envs,
            )
            print(
                f"\n[DICE SWEEP] [{index}/{len(checkpoints)}] {checkpoint.name}",
                flush=True,
            )
            if reusable and not args.force:
                print(
                    f"[DICE SWEEP] Reusing completed evaluation: {summary_path}",
                    flush=True,
                )
                state["reused"].append(checkpoint.name)
            else:
                state["active_checkpoint"] = checkpoint.name
                state_path.write_text(json.dumps(state, indent=2))
                command = [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts" / "evaluate_rsl.py"),
                    "--task",
                    "DICE-Shadow-Eval-v0",
                    "--checkpoint",
                    str(checkpoint),
                    "--episodes",
                    str(args.episodes),
                    "--num_envs",
                    str(args.num_envs),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--headless",
                    "--output",
                    str(output_directory),
                ]
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                state["completed"].append(checkpoint.name)
                state["active_checkpoint"] = None
            if not summary_path.is_file():
                raise RuntimeError(
                    f"Evaluation did not produce expected summary: {summary_path}"
                )
            summary_paths.append(summary_path)
            state_path.write_text(json.dumps(state, indent=2))

        payload, table = write_ranking_artifacts(
            summary_paths,
            sweep_directory,
            success_threshold=args.success_threshold,
        )
        state.update(
            {
                "status": "complete",
                "finished_at": datetime.now().astimezone().isoformat(),
                "selected_checkpoint": payload["selected_checkpoint"],
            }
        )
        state_path.write_text(json.dumps(state, indent=2))
        write_artifact_manifest(run_directory)

        print("\n[DICE SWEEP] Checkpoint ranking", flush=True)
        print(table, flush=True)
        print(
            f"\n[DICE SWEEP] Selected checkpoint: {payload['selected_checkpoint']}",
            flush=True,
        )
        print(f"[DICE SWEEP] Results: {sweep_directory}", flush=True)
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
