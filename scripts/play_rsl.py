"""Capture deterministic DICE presentation trajectories with RSL-RL."""

# Isaac Sim must launch before importing its runtime-dependent modules.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Capture or replay a deterministic DICE presentation trajectory.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--condition",
    choices=("nominal", "robust", "adverse"),
    default="nominal",
)
parser.add_argument(
    "--task",
    default=None,
    help="Advanced task override; the condition selects a presentation task by default.",
)
parser.add_argument("--output", default="videos/DICE")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Maximum policy steps; defaults to the selected condition horizon.",
)
parser.add_argument(
    "--command-limit",
    type=int,
    default=None,
    help=(
        "Successful-command termination target; zero disables command-limit "
        "termination and the task horizon/drop decides the outcome."
    ),
)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--camera", choices=("hero", "top", "side"), default="hero")
parser.add_argument("--resolution", default="1920x1080")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--trajectory-input", default=None)
parser.add_argument("--trajectory-output", default=None)
parser.add_argument("--no_video", action="store_true")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.seed < 0:
    parser.error("--seed must be non-negative")
if args.video_length is not None and args.video_length <= 0:
    parser.error("--video_length must be positive")
if args.command_limit is not None and args.command_limit < 0:
    parser.error("--command-limit must be non-negative")
if args.fps <= 0:
    parser.error("--fps must be positive")
if not args.no_video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401
from dicedial.agents.rsl_rl_ppo_cfg import (
    compatible_checkpoint_path,
    make_runner_cfg,
)
from dicedial.portfolio_video import (
    CAMERA_PRESETS,
    PORTFOLIO_CONDITIONS,
    PORTFOLIO_SCHEMA_VERSION,
    PRESENTATION_COLLISION_EXTENT_M,
    parse_resolution,
    sha256_file,
    write_json,
)


METRIC_FIELDS = (
    "frame_index",
    "step",
    "sim_time_seconds",
    "target_face",
    "top_face",
    "alignment",
    "angular_error_degrees",
    "position_error",
    "hold_progress",
    "success",
    "completed_face",
    "success_latency_seconds",
    "commands_completed",
    "drop",
    "done",
    "status",
    "reward",
)


def _first(mapping, key, default=0):
    value = mapping.get(key)
    if value is None:
        return default
    if hasattr(value, "detach"):
        return value[0].detach().cpu().item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _task_metrics_snapshot(unwrapped):
    metrics = unwrapped.get_task_metrics()
    alignment = float(_first(metrics, "alignment", 0.0))
    return {
        "step": -1,
        "sim_time_seconds": 0.0,
        "target_face": int(_first(metrics, "target_face", 0)),
        "top_face": int(_first(metrics, "top_face", 0)),
        "alignment": alignment,
        "angular_error_degrees": math.degrees(
            math.acos(max(-1.0, min(1.0, alignment)))
        ),
        "position_error": float(_first(metrics, "position_error", 0.0)),
        "hold_progress": float(_first(metrics, "hold_progress", 0.0)),
        "success": 0,
        "completed_face": 0,
        "success_latency_seconds": 0.0,
        "commands_completed": int(_first(metrics, "commands_completed", 0)),
        "drop": int(bool(_first(metrics, "out_of_reach", False))),
        "done": 0,
        "status": "rotating",
        "reward": 0.0,
    }


def _physics_snapshot(unwrapped, presentation_task):
    view = unwrapped.object.root_physx_view
    payload = {
        "configured_collision_extent_m": (
            list(PRESENTATION_COLLISION_EXTENT_M) if presentation_task else None
        ),
    }
    for key, method_name in (
        ("mass", "get_masses"),
        ("inertia", "get_inertias"),
        ("center_of_mass", "get_coms"),
    ):
        method = getattr(view, method_name, None)
        if method is None:
            payload[key] = None
            continue
        try:
            value = method().detach().cpu()
            if value.ndim > 1:
                value = value[0]
            payload[key] = value.reshape(-1).tolist()
        except Exception as exc:  # simulator API compatibility diagnostic
            payload[key] = None
            payload[f"{key}_error"] = str(exc)
    return payload


def _write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_trajectory(path, checkpoint_hash, condition, task, seed, command_limit):
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Trajectory checkpoint hash does not match --checkpoint")
    if metadata["condition"] != condition:
        raise ValueError("Trajectory condition does not match --condition")
    if metadata["task"] != task:
        raise ValueError("Trajectory task does not match the resolved task")
    if int(metadata["seed"]) != int(seed):
        raise ValueError("Trajectory seed does not match --seed")
    # Presentation trajectories written before this field was introduced used
    # the environment defaults: six commands for nominal/robust, no limit for
    # adverse. Preserve those valid low-level replays while still rejecting an
    # old six-command trace when the new 12-command contract is requested.
    legacy_command_limit = 0 if condition == "adverse" else 6
    recorded_command_limit = int(metadata.get("command_limit", legacy_command_limit))
    if recorded_command_limit != int(command_limit):
        raise ValueError("Trajectory command limit does not match --command-limit")
    if actions.ndim != 2 or actions.shape[1] != 20:
        raise ValueError(
            f"Expected trajectory actions shaped [steps, 20], got {actions.shape}"
        )
    return path, actions, metadata


def _save_trajectory(path, actions, metadata):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 20:
        raise ValueError(f"Cannot save invalid action trajectory shape: {array.shape}")
    np.savez_compressed(path, actions=array, metadata=np.array(json.dumps(metadata)))
    return path


def main():
    condition = PORTFOLIO_CONDITIONS[args.condition]
    task = args.task or condition["task"]
    max_steps = args.video_length or int(condition["max_steps"])
    resolution = parse_resolution(args.resolution)
    camera = CAMERA_PRESETS[args.camera]
    device = args.device or "cuda:0"

    checkpoint_source = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_source}")
    checkpoint_hash = sha256_file(checkpoint_source)

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "capture_summary.json"
    initial_metrics_path = output_dir / "initial_metrics.json"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Capture output already exists: {output_dir}. Use a new directory."
        )

    raw_directory = output_dir / "raw"
    if not args.no_video:
        if raw_directory.exists() and any(raw_directory.iterdir()):
            raise FileExistsError(f"Raw video directory is not empty: {raw_directory}")

    agent_cfg = make_runner_cfg(seed=args.seed, device=device)
    env_cfg = parse_env_cfg(
        task,
        device=device,
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    if args.command_limit is not None:
        env_cfg.max_commands_per_episode = args.command_limit
    command_limit = int(env_cfg.max_commands_per_episode)
    env_cfg.emit_step_metrics = True
    env_cfg.wait_for_textures = not args.no_video
    # Every presentation/audit invocation uses one environment. Disabling the
    # clone-in-Fabric path avoids Isaac's misleading "Failed to clone in
    # Fabric" diagnostic when the stock 256-environment evaluation config is
    # deliberately overridden to one instance.
    env_cfg.scene.clone_in_fabric = False
    env_cfg.viewer.eye = camera["eye"]
    env_cfg.viewer.lookat = camera["lookat"]
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.resolution = resolution

    base_env = gym.make(
        task,
        cfg=env_cfg,
        render_mode=None if args.no_video else "rgb_array",
    )
    wrapped_env = base_env
    if not args.no_video:
        wrapped_env = gym.wrappers.RecordVideo(
            wrapped_env,
            video_folder=str(raw_directory),
            step_trigger=lambda step: step == 0,
            video_length=max_steps + 1,
            name_prefix=f"DICE-{args.condition}-{args.camera}-seed-{args.seed}",
            fps=args.fps,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(wrapped_env, clip_actions=agent_cfg.clip_actions)
    checkpoint = compatible_checkpoint_path(checkpoint_source)
    observations = env.get_observations()
    initial_metrics = _task_metrics_snapshot(env.unwrapped)
    initial_metrics["frame_index"] = 0
    write_json(initial_metrics_path, initial_metrics)
    physics_snapshot = _physics_snapshot(
        env.unwrapped, task in {item["task"] for item in PORTFOLIO_CONDITIONS.values()}
    )

    replay_actions = None
    trajectory_source = None
    trajectory_metadata = None
    policy = None
    if args.trajectory_input:
        trajectory_source, replay_actions, trajectory_metadata = _load_trajectory(
            args.trajectory_input,
            checkpoint_hash,
            args.condition,
            task,
            args.seed,
            command_limit,
        )
        max_steps = min(max_steps, len(replay_actions))
    else:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    rows = []
    recorded_actions = []
    terminal_done = False
    try:
        from tqdm import tqdm

        progress = tqdm(
            total=max_steps,
            desc=f"[DICE {args.condition}/{args.camera}]",
            unit="step",
            dynamic_ncols=True,
        )
    except ImportError:
        progress = None

    step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))
    for step in range(max_steps):
        with torch.inference_mode():
            if replay_actions is None:
                actions = policy(observations)
                recorded_actions.append(actions[0].detach().cpu().numpy())
            else:
                actions = torch.as_tensor(
                    replay_actions[step],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
            observations, rewards, dones, extras = env.step(actions)

        alignment = float(_first(extras, "dice_alignment", 0.0))
        success = bool(_first(extras, "dice_success", False))
        drop = bool(_first(extras, "dice_drop", False))
        done = bool(dones[0].item())
        hold = float(_first(extras, "dice_hold_progress", 0.0))
        if drop:
            status = "dropped"
        elif success:
            status = "confirmed"
        elif hold > 0.0:
            status = "holding"
        else:
            status = "rotating"
        row = {
            "frame_index": step,
            "step": step,
            "sim_time_seconds": (step + 1) * step_dt,
            "target_face": int(_first(extras, "dice_target_face", 0)),
            "top_face": int(_first(extras, "dice_top_face", 0)),
            "alignment": alignment,
            "angular_error_degrees": math.degrees(
                math.acos(max(-1.0, min(1.0, alignment)))
            ),
            "position_error": float(_first(extras, "dice_position_error", 0.0)),
            "hold_progress": hold,
            "success": int(success),
            "completed_face": int(_first(extras, "dice_completed_face", 0)),
            "success_latency_seconds": float(
                _first(extras, "dice_success_latency_steps", 0.0)
            )
            * step_dt,
            "commands_completed": int(_first(extras, "dice_commands_completed", 0)),
            "drop": int(drop),
            "done": int(done),
            "status": status,
            "reward": float(rewards[0].detach().cpu().item()),
        }
        rows.append(row)

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {
                    "Target": row["target_face"],
                    "Top": row["top_face"],
                    "Cmds": row["commands_completed"],
                    "State": status,
                }
            )

        if policy is not None and hasattr(policy, "reset"):
            policy.reset(dones)
        if done:
            terminal_done = True
            break

    if progress is not None:
        progress.close()

    _write_metrics(metrics_path, rows)
    env.close()

    raw_videos = sorted(raw_directory.glob("*.mp4")) if not args.no_video else []
    if not args.no_video and len(raw_videos) != 1:
        raise RuntimeError(
            f"Expected exactly one raw MP4 in {raw_directory}, found {len(raw_videos)}"
        )

    if rows:
        final_row = rows[-1]
        commands_completed = int(final_row["commands_completed"])
        dropped = bool(final_row["drop"])
        duration_seconds = float(final_row["sim_time_seconds"])
    else:
        commands_completed = 0
        dropped = False
        duration_seconds = 0.0

    if dropped:
        outcome = "dropped"
    elif command_limit > 0 and commands_completed >= command_limit:
        outcome = "completed_sequence"
    elif terminal_done:
        outcome = "timeout"
    elif replay_actions is not None and len(rows) == len(replay_actions):
        outcome = "trajectory_end"
    else:
        outcome = "step_limit"

    if args.trajectory_output and replay_actions is None:
        trajectory_metadata = {
            "schema_version": PORTFOLIO_SCHEMA_VERSION,
            "checkpoint": checkpoint_source.name,
            "checkpoint_sha256": checkpoint_hash,
            "condition": args.condition,
            "task": task,
            "seed": args.seed,
            "command_limit": command_limit,
            "step_dt_seconds": step_dt,
            "steps": len(recorded_actions),
        }
        trajectory_source = _save_trajectory(
            args.trajectory_output, recorded_actions, trajectory_metadata
        )

    summary = {
        "schema_version": PORTFOLIO_SCHEMA_VERSION,
        "project": "DICE",
        "status": "complete",
        "condition": args.condition,
        "condition_definition": condition,
        "task": task,
        "seed": args.seed,
        "command_limit": command_limit,
        "camera": args.camera,
        "camera_definition": camera,
        "resolution": list(resolution),
        "raw_fps": args.fps,
        # RecordVideo is started by the first step trigger, so every captured
        # source frame represents a post-step state (there is no reset frame).
        "recording_frame_origin": "post_step",
        "checkpoint": str(checkpoint_source),
        "checkpoint_loaded": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "trajectory": str(trajectory_source) if trajectory_source else None,
        "trajectory_replay": replay_actions is not None,
        "trajectory_metadata": trajectory_metadata,
        "metrics": str(metrics_path),
        "initial_metrics": str(initial_metrics_path),
        "raw_video": str(raw_videos[0]) if raw_videos else None,
        "steps": len(rows),
        "duration_seconds": duration_seconds,
        "commands_completed": commands_completed,
        "dropped": dropped,
        "outcome": outcome,
        "terminal_done": terminal_done,
        "physics_snapshot": physics_snapshot,
    }
    write_json(summary_path, summary)

    print(f"[DICE] Checkpoint : {checkpoint_source}")
    print(f"[DICE] Condition  : {args.condition}")
    print(f"[DICE] Camera     : {args.camera}")
    print(f"[DICE] Outcome    : {outcome}")
    print(f"[DICE] Commands   : {commands_completed}")
    print(f"[DICE] Metrics    : {metrics_path}")
    print(f"[DICE] Summary    : {summary_path}")
    if raw_videos:
        print(f"[DICE] Raw video : {raw_videos[0]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
