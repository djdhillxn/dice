"""Launch a small vectorized simulation and verify the task contract."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="DiceDial Isaac Lab smoke test.")
parser.add_argument("--task", default="DiceDial-Shadow-Random-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device or "cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(args.task, cfg=env_cfg)
    observations, _ = env.reset()

    policy_observation = observations["policy"]
    assert policy_observation.shape == (args.num_envs, 164)
    assert torch.isfinite(policy_observation).all()

    for _ in range(args.steps):
        actions = 2.0 * torch.rand(args.num_envs, 20, device=env.unwrapped.device) - 1.0
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert observations["policy"].shape == (args.num_envs, 164)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert terminated.shape == (args.num_envs,)
        assert truncated.shape == (args.num_envs,)

    metrics = env.unwrapped.get_task_metrics()
    assert metrics["target_face"].min().item() >= 1
    assert metrics["target_face"].max().item() <= 6
    assert torch.isfinite(metrics["alignment"]).all()

    env.close()
    print("DiceDial smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
