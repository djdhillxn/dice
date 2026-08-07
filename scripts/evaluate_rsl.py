"""Evaluate an RSL-RL DICE checkpoint on nominal or robust physics."""

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
parser.add_argument("--output", default="evaluation/nominal")
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
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = make_runner_cfg(seed=args.seed, device=device)
    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    checkpoint = compatible_checkpoint_path(Path(args.checkpoint).expanduser().resolve())
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
    committed_latencies = []
    per_face_successes = np.zeros(6, dtype=np.int64)
    per_face_attempts = np.zeros(6, dtype=np.int64)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.episodes, desc="[DICE Evaluation]", unit="ep", dynamic_ncols=True)
    except ImportError:
        pbar = None

    last_ep_count = 0

    while len(episode_rows) < args.episodes:
        with torch.inference_mode():
            actions = policy(observations)
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
            unfinished_command = not final_step_success
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
            pbar.set_postfix({
                "MeanCmds": f"{total_successes / max(len(episode_rows), 1):.2f}",
                "DropRate": f"{total_drops / max(len(episode_rows), 1):.3f}",
            })

        if hasattr(policy, "reset"):
            policy.reset(dones)

    if pbar is not None:
        pbar.close()

    frame = pd.DataFrame(episode_rows)
    frame.to_csv(output_dir / "episodes.csv", index=False)

    attempted_commands = int(per_face_attempts.sum())
    control_dt = float(env.unwrapped.step_dt)
    latency_array = np.asarray(committed_latencies, dtype=np.float64)

    summary = {
        "project": "DICE",
        "task": args.task,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "episodes": int(len(frame)),
        "successful_commands": int(total_successes),
        "attempted_commands": attempted_commands,
        "target_face_success_rate": float(total_successes / max(attempted_commands, 1)),
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
        "control_dt_seconds": control_dt,
        "per_face": {
            str(face): {
                "successes": int(per_face_successes[face - 1]),
                "attempts": int(per_face_attempts[face - 1]),
                "success_rate": float(
                    per_face_successes[face - 1]
                    / max(per_face_attempts[face - 1], 1)
                ),
            }
            for face in range(1, 7)
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
