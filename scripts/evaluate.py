"""Evaluate a trained DiceDial policy and save defensible task metrics."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate DiceDial.")
parser.add_argument("--task", default="DiceDial-Shadow-Sequence-v0")
parser.add_argument("--model", required=True)
parser.add_argument("--vecnormalize", required=True)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--output", default="evaluation")
parser.add_argument("--stochastic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401


def _scalar(info, key, default=0.0):
    value = info.get(key, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device or "cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed

    raw_env = gym.make(args.task, cfg=env_cfg)
    vec_env = Sb3VecEnvWrapper(raw_env, fast_variant=False)
    env = VecNormalize.load(args.vecnormalize, vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model, env=env, device=args.device or "cuda:0")
    observation = env.reset()

    episode_rows = []
    total_successes = 0
    total_drops = 0
    committed_latencies = []
    per_face_successes = np.zeros(6, dtype=np.int64)
    per_face_attempts = np.zeros(6, dtype=np.int64)

    # Successes are committed only when their episode finishes. This avoids
    # contaminating the requested episode count with partial vector episodes.
    pending_successes = np.zeros(args.num_envs, dtype=np.int64)
    pending_face_successes = np.zeros((args.num_envs, 6), dtype=np.int64)
    pending_latencies = [[] for _ in range(args.num_envs)]

    while len(episode_rows) < args.episodes:
        actions, _ = model.predict(observation, deterministic=not args.stochastic)
        observation, _, dones, infos = env.step(actions)

        for index, info in enumerate(infos):
            success_this_step = bool(_scalar(info, "dicedial_success", False))
            if success_this_step:
                completed_face = int(_scalar(info, "dicedial_completed_face", 0))
                pending_successes[index] += 1
                if 1 <= completed_face <= 6:
                    pending_face_successes[index, completed_face - 1] += 1
                latency = float(_scalar(info, "dicedial_success_latency_steps", 0.0))
                if latency > 0:
                    pending_latencies[index].append(latency)

            if not dones[index]:
                continue

            dropped = bool(_scalar(info, "dicedial_drop", False))
            time_limit = bool(info.get("TimeLimit.truncated", False))
            commands_completed = int(pending_successes[index])
            configured_limit = int(getattr(env_cfg, "max_commands_per_episode", 0))
            command_limit_reached = configured_limit > 0 and commands_completed >= configured_limit
            unfinished_command = not success_this_step and not command_limit_reached

            total_successes += commands_completed
            total_drops += int(dropped)
            per_face_successes += pending_face_successes[index]
            per_face_attempts += pending_face_successes[index]
            committed_latencies.extend(pending_latencies[index])

            active_face = int(_scalar(info, "dicedial_target_face", 0))
            if unfinished_command and 1 <= active_face <= 6:
                per_face_attempts[active_face - 1] += 1

            episode_rows.append(
                {
                    "episode": len(episode_rows),
                    "commands_completed": commands_completed,
                    "dropped": dropped,
                    "time_limit": time_limit,
                    "command_limit_reached": command_limit_reached,
                    "final_step_success": success_this_step,
                    "unfinished_command": unfinished_command,
                    "active_face_at_end": active_face,
                    "episode_return": info.get("episode", {}).get("r") if info.get("episode") else None,
                    "episode_length": info.get("episode", {}).get("l") if info.get("episode") else None,
                }
            )

            pending_successes[index] = 0
            pending_face_successes[index] = 0
            pending_latencies[index] = []

            if len(episode_rows) >= args.episodes:
                break

    frame = pd.DataFrame(episode_rows)
    frame.to_csv(output_dir / "episodes.csv", index=False)

    attempted_commands = int(per_face_attempts.sum())
    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    latency_array = np.asarray(committed_latencies, dtype=np.float64)
    summary = {
        "task": args.task,
        "seed": args.seed,
        "episodes": int(len(frame)),
        "successful_commands": int(total_successes),
        "attempted_commands": attempted_commands,
        "target_face_success_rate": float(total_successes / max(attempted_commands, 1)),
        "median_time_to_target_steps": float(np.median(latency_array)) if latency_array.size else None,
        "median_time_to_target_seconds": float(np.median(latency_array) * control_dt) if latency_array.size else None,
        "drop_rate": float(total_drops / max(len(frame), 1)),
        "mean_consecutive_commands": float(frame["commands_completed"].mean()),
        "median_consecutive_commands": float(frame["commands_completed"].median()),
        "max_consecutive_commands": int(frame["commands_completed"].max()),
        "deterministic_policy": not args.stochastic,
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


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
