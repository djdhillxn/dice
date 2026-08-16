"""Evaluate an RSL-RL DICE checkpoint under one registered physics condition."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate a DICE RSL-RL checkpoint.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--task", default="DICE-Shadow-Eval-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument(
    "--output",
    default=None,
    help="Output directory. Defaults to <checkpoint-run>/evaluation/<task-kind>.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401
from dicedial.agents.rsl_rl_ppo_cfg import (
    compatible_checkpoint_path,
    make_runner_cfg,
)
from dicedial.final_evaluation import sha256_file
from dicedial.training_artifacts import write_artifact_manifest


def _numpy_tensor(extras, key, fallback, dtype=None):
    value = extras.get(key)
    if value is None:
        array = np.asarray(fallback)
    elif hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def main():
    device = args.device or "cuda:0"
    checkpoint_source = Path(args.checkpoint).expanduser().resolve()
    checkpoint = compatible_checkpoint_path(checkpoint_source)
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        if "Robust" in args.task:
            task_kind = "robust"
        elif "Adverse" in args.task:
            task_kind = "adverse"
        else:
            task_kind = "nominal"
        output_dir = checkpoint_source.parent / "evaluation" / task_kind
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = make_runner_cfg(seed=args.seed, device=device)
    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    # Config dictionaries contain callable event terms on Isaac Lab releases
    # supported by this repository. Normalize those values now so the summary
    # remains valid JSON while retaining the exact resolved event parameters.
    resolved_environment = json.loads(json.dumps(env_cfg.to_dict(), default=str))

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    observations = env.get_observations()
    num_envs = int(env.num_envs)

    pending_successes = np.zeros(num_envs, dtype=np.int64)
    pending_face_successes = np.zeros((num_envs, 6), dtype=np.int64)
    pending_latencies = [[] for _ in range(num_envs)]
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)

    episode_rows = []
    total_successes = 0
    total_drops = 0
    action_abs_sum = 0.0
    action_square_sum = 0.0
    action_out_of_bounds_count = 0
    action_value_count = 0
    committed_latencies = []
    per_face_successes = np.zeros(6, dtype=np.int64)
    per_face_attempts = np.zeros(6, dtype=np.int64)

    try:
        from tqdm import tqdm

        pbar = tqdm(
            total=args.episodes, desc="[DICE Evaluation]", unit="ep", dynamic_ncols=True
        )
    except ImportError:
        pbar = None

    last_ep_count = 0

    while len(episode_rows) < args.episodes:
        with torch.inference_mode():
            actions = policy(observations)
            action_abs_sum += float(actions.abs().sum().item())
            action_square_sum += float(actions.square().sum().item())
            action_out_of_bounds_count += int((actions.abs() >= 1.0).sum().item())
            action_value_count += actions.numel()
            observations, rewards, dones, extras = env.step(actions)

        reward_array = rewards.detach().cpu().numpy()
        done_array = dones.detach().cpu().numpy().astype(bool)
        success_array = _numpy_tensor(
            extras,
            "dice_success",
            np.zeros(num_envs, dtype=bool),
            bool,
        )
        completed_faces = _numpy_tensor(
            extras,
            "dice_completed_face",
            np.zeros(num_envs),
            np.int64,
        )
        latency_steps = _numpy_tensor(
            extras,
            "dice_success_latency_steps",
            np.zeros(num_envs),
            np.float64,
        )
        target_faces = _numpy_tensor(
            extras,
            "dice_target_face",
            np.zeros(num_envs),
            np.int64,
        )
        drop_array = _numpy_tensor(
            extras,
            "dice_drop",
            np.zeros(num_envs, dtype=bool),
            bool,
        )
        time_outs = _numpy_tensor(
            extras,
            "time_outs",
            np.zeros(num_envs, dtype=bool),
            bool,
        )

        episode_returns += reward_array
        episode_lengths += 1

        success_indices = np.flatnonzero(success_array)
        for env_index in success_indices:
            face = int(completed_faces[env_index])
            pending_successes[env_index] += 1
            if 1 <= face <= 6:
                pending_face_successes[env_index, face - 1] += 1
            latency = float(latency_steps[env_index])
            if latency > 0:
                pending_latencies[env_index].append(latency)

        done_indices = np.flatnonzero(done_array)
        for env_index in done_indices:
            if len(episode_rows) >= args.episodes:
                break

            commands_completed = int(pending_successes[env_index])
            dropped = bool(drop_array[env_index])
            time_limit = bool(time_outs[env_index])
            final_step_success = bool(success_array[env_index])

            total_successes += commands_completed
            total_drops += int(dropped)
            per_face_successes += pending_face_successes[env_index]
            per_face_attempts += pending_face_successes[env_index]
            committed_latencies.extend(pending_latencies[env_index])

            active_face = int(target_faces[env_index])
            # Training/evaluation immediately issues another target after a
            # completion. Therefore an episode that ends on a success event
            # still has a newly-issued unfinished command.
            unfinished_command = True
            if unfinished_command and 1 <= active_face <= 6:
                per_face_attempts[active_face - 1] += 1

            episode_rows.append(
                {
                    "episode": len(episode_rows),
                    "commands_completed": commands_completed,
                    "dropped": dropped,
                    "time_limit": time_limit,
                    "final_step_success": final_step_success,
                    "unfinished_command": unfinished_command,
                    "active_face_at_end": active_face,
                    "episode_return": float(episode_returns[env_index]),
                    "episode_length": int(episode_lengths[env_index]),
                }
            )

            pending_successes[env_index] = 0
            pending_face_successes[env_index] = 0
            pending_latencies[env_index] = []
            episode_returns[env_index] = 0.0
            episode_lengths[env_index] = 0

        if pbar is not None and len(episode_rows) > last_ep_count:
            added = len(episode_rows) - last_ep_count
            last_ep_count = len(episode_rows)
            pbar.update(added)
            pbar.set_postfix(
                {
                    "MeanCmds": f"{total_successes / max(len(episode_rows), 1):.2f}",
                    "DropRate": f"{total_drops / max(len(episode_rows), 1):.3f}",
                }
            )

        if hasattr(policy, "reset"):
            policy.reset(dones)

    if pbar is not None:
        pbar.close()

    frame = pd.DataFrame(episode_rows)
    frame.to_csv(output_dir / "episodes.csv", index=False)

    attempted_commands = int(per_face_attempts.sum())
    control_dt = float(env.unwrapped.step_dt)
    latency_array = np.asarray(committed_latencies, dtype=np.float64)
    total_sim_seconds = float(frame["episode_length"].sum() * control_dt)
    episode_any_completion_fraction = float((frame["commands_completed"] >= 1).mean())

    policy_module = runner.alg.policy
    if hasattr(policy_module, "log_std"):
        checkpoint_noise = policy_module.log_std.detach().exp()
    elif hasattr(policy_module, "std"):
        checkpoint_noise = policy_module.std.detach()
    else:
        checkpoint_noise = None

    summary = {
        "project": "DICE",
        "task": args.task,
        "checkpoint": str(checkpoint),
        "checkpoint_source": str(checkpoint_source),
        "checkpoint_sha256": sha256_file(checkpoint_source),
        "seed": args.seed,
        "num_envs": num_envs,
        "episodes": int(len(frame)),
        "environment_config": {
            "class": type(env_cfg).__name__,
            "events": resolved_environment.get("events"),
            "episode_length_s": float(env_cfg.episode_length_s),
            "control_decimation": int(env_cfg.decimation),
        },
        "successful_commands": int(total_successes),
        "attempted_commands": attempted_commands,
        "issued_command_completion_rate": float(
            total_successes / max(attempted_commands, 1)
        ),
        "target_face_success_rate": float(total_successes / max(attempted_commands, 1)),
        "episode_any_completion_fraction": episode_any_completion_fraction,
        "zero_completion_episode_fraction": 1.0 - episode_any_completion_fraction,
        "completed_commands_per_sim_minute": float(
            total_successes / max(total_sim_seconds / 60.0, 1.0e-8)
        ),
        "median_time_to_target_steps": (
            float(np.median(latency_array)) if latency_array.size else None
        ),
        "median_time_to_target_seconds": (
            float(np.median(latency_array) * control_dt) if latency_array.size else None
        ),
        "drop_rate": float(total_drops / max(len(frame), 1)),
        "mean_consecutive_commands": float(frame["commands_completed"].mean()),
        "median_consecutive_commands": float(frame["commands_completed"].median()),
        "max_consecutive_commands": int(frame["commands_completed"].max()),
        "deterministic_action_mean_abs": float(
            action_abs_sum / max(action_value_count, 1)
        ),
        "deterministic_action_rms": float(
            np.sqrt(action_square_sum / max(action_value_count, 1))
        ),
        "deterministic_action_out_of_bounds_fraction": float(
            action_out_of_bounds_count / max(action_value_count, 1)
        ),
        "checkpoint_noise_std": (
            {
                "minimum": float(checkpoint_noise.min().item()),
                "mean": float(checkpoint_noise.mean().item()),
                "maximum": float(checkpoint_noise.max().item()),
            }
            if checkpoint_noise is not None
            else None
        ),
        "control_dt_seconds": control_dt,
        "per_face": {
            str(face): {
                "successes": int(per_face_successes[face - 1]),
                "attempts": int(per_face_attempts[face - 1]),
                "success_rate": float(
                    per_face_successes[face - 1] / max(per_face_attempts[face - 1], 1)
                ),
            }
            for face in range(1, 7)
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    env.close()

    # Evaluation normally lives under the checkpoint run. Refresh that run's
    # transfer manifest so it does not silently omit post-training artifacts.
    checkpoint_run_dir = checkpoint_source.parent
    try:
        output_dir.relative_to(checkpoint_run_dir)
    except ValueError:
        pass
    else:
        write_artifact_manifest(checkpoint_run_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
