"""Render a deterministic six-command DiceDial demonstration."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Render DiceDial policy video.")
parser.add_argument("--task", default="DiceDial-Shadow-Play-v0")
parser.add_argument("--model", required=True)
parser.add_argument("--vecnormalize", required=True)
parser.add_argument("--output", default="videos/dicedial")
parser.add_argument("--video_length", type=int, default=2400)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--no_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.no_video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401


def main():
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device or "cuda:0",
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed

    raw_env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if not args.no_video else None,
    )
    wrapped_env = raw_env
    if not args.no_video:
        wrapped_env = gym.wrappers.RecordVideo(
            wrapped_env,
            video_folder=str(output_dir / "raw"),
            episode_trigger=lambda episode: episode == 0,
            video_length=args.video_length,
            name_prefix="dicedial",
            disable_logger=True,
        )

    vec_env = Sb3VecEnvWrapper(wrapped_env, fast_variant=False)
    env = VecNormalize.load(args.vecnormalize, vec_env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.model, env=env, device=args.device or "cuda:0")

    observation = env.reset()
    rows = []

    for step in range(args.video_length):
        actions, _ = model.predict(observation, deterministic=not args.stochastic)
        observation, rewards, dones, _ = env.step(actions)
        metrics = raw_env.unwrapped.get_task_metrics()
        rows.append(
            {
                "step": step,
                "target_face": int(metrics["target_face"][0].item()),
                "top_face": int(metrics["top_face"][0].item()),
                "alignment": float(metrics["alignment"][0].item()),
                "position_error": float(metrics["position_error"][0].item()),
                "hold_progress": float(metrics["hold_progress"][0].item()),
                "commands_completed": int(metrics["commands_completed"][0].item()),
                "reward": float(rewards[0]),
            }
        )
        if int(metrics["commands_completed"][0].item()) >= len(env_cfg.target_sequence):
            break
        if dones[0]:
            break

    pd.DataFrame(rows).to_csv(output_dir / "video_metrics.csv", index=False)
    env.close()
    print("[DiceDial] Raw video folder:", output_dir / "raw")
    print("[DiceDial] Overlay data:", output_dir / "video_metrics.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
