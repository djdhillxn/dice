"""Train DiceDial with Stable-Baselines3 PPO."""

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train DiceDial with SB3 PPO.")
parser.add_argument("--task", default="DiceDial-Shadow-Sequence-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_timesteps", type=int, default=None)
parser.add_argument("--output_root", default="outputs")
parser.add_argument("--run_name", default=None)
parser.add_argument("--checkpoint", default=None, help="SB3 model.zip to resume or warm-start from.")
parser.add_argument("--vecnormalize", default=None, help="VecNormalize .pkl paired with the checkpoint.")
parser.add_argument("--checkpoint_freq", type=int, default=1000, help="Frequency in vector steps.")
parser.add_argument("--metrics_freq", type=int, default=1000, help="Frequency in vector steps.")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_interval", type=int, default=250000)
parser.add_argument("--video_length", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import dicedial.tasks  # noqa: F401
from dicedial.callbacks import DiceDialMetricsCallback


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or timestamp
    output_dir = Path(args.output_root).resolve() / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv))

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device or "cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(output_dir)

    agent_cfg = load_cfg_from_registry(args.task, "sb3_cfg_entry_point")
    agent_cfg["seed"] = args.seed
    if args.total_timesteps is not None:
        agent_cfg["n_timesteps"] = int(args.total_timesteps)

    dump_yaml(str(output_dir / "env.yaml"), env_cfg)
    dump_yaml(str(output_dir / "agent.yaml"), agent_cfg)

    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_name = agent_cfg.pop("policy")
    total_timesteps = int(agent_cfg.pop("n_timesteps"))

    raw_env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args.video else None,
    )

    wrapped_env = raw_env
    if args.video:
        video_kwargs = {
            "video_folder": str(output_dir / "videos" / "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        wrapped_env = gym.wrappers.RecordVideo(wrapped_env, **video_kwargs)

    env = Sb3VecEnvWrapper(wrapped_env, fast_variant=True)

    normalization = {}
    for key in ("normalize_input", "normalize_value", "clip_obs"):
        if key in agent_cfg:
            normalization[key] = agent_cfg.pop(key)

    if args.vecnormalize:
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = True
        env.norm_reward = normalization.get("normalize_value", True)
    elif normalization.get("normalize_input", False):
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=normalization.get("normalize_value", False),
            clip_obs=normalization.get("clip_obs", 10.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    if args.checkpoint:
        model = PPO.load(args.checkpoint, env=env, device=agent_cfg.get("device", "cuda:0"), print_system_info=True)
        model.tensorboard_log = str(output_dir)
    else:
        model = PPO(
            policy_name,
            env,
            tensorboard_log=str(output_dir),
            **agent_cfg,
        )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq, 1),
                save_path=str(output_dir / "checkpoints"),
                name_prefix="dicedial",
                save_replay_buffer=False,
                save_vecnormalize=True,
                verbose=2,
            ),
            DiceDialMetricsCallback(
                raw_env,
                output_dir,
                sample_every=args.metrics_freq,
            ),
        ]
    )

    metadata = {
        "task": args.task,
        "num_envs": env_cfg.scene.num_envs,
        "seed": args.seed,
        "total_timesteps": total_timesteps,
        "checkpoint": args.checkpoint,
        "vecnormalize": args.vecnormalize,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))

    print("[DiceDial] Output:", output_dir)
    print("[DiceDial] Environments:", env_cfg.scene.num_envs)
    print("[DiceDial] Total timesteps:", total_timesteps)

    with contextlib.suppress(KeyboardInterrupt):
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.checkpoint),
            log_interval=None,
        )

    model.save(str(output_dir / "model"))
    if isinstance(env, VecNormalize):
        env.save(str(output_dir / "model_vecnormalize.pkl"))

    env.close()
    print("[DiceDial] Saved:", output_dir / "model.zip")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
