"""Train DICE once on the complete final task with RSL-RL PPO."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Train DICE with RSL-RL PPO.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--task", default="DICE-Shadow-Train-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=10_000)
parser.add_argument("--run_name", default="final")
parser.add_argument("--output_root", default="outputs/DICE")
parser.add_argument("--resume", default=None, help="RSL-RL .pt checkpoint")
parser.add_argument("--seed", type=int, default=42)
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
import torch
from rsl_rl.runners import OnPolicyRunner

# Match Isaac Lab's official RSL-RL training path.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import dicedial.tasks  # noqa: F401
from dicedial.agents.rsl_rl_ppo_cfg import (
    compatible_checkpoint_path,
    make_runner_cfg,
)


def main():
    device = args.device or "cuda:0"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory_name = timestamp if not args.run_name else f"{timestamp}_{args.run_name}"
    output_dir = Path(args.output_root).resolve() / directory_name
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = make_runner_cfg(
        seed=args.seed,
        device=device,
        max_iterations=args.max_iterations,
        run_name=args.run_name,
    )

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    env_cfg.log_dir = str(output_dir)

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=str(output_dir),
        device=agent_cfg.device,
    )

    if args.resume:
        checkpoint = compatible_checkpoint_path(Path(args.resume).expanduser().resolve())
        runner.load(checkpoint)
        print(f"[DICE] Resumed from: {checkpoint}")

    completed_iterations = int(getattr(runner, "current_learning_iteration", 0))
    remaining_iterations = max(args.max_iterations - completed_iterations, 0)

    metadata = {
        "project": "DICE",
        "task": args.task,
        "seed": args.seed,
        "device": device,
        "num_envs": int(env_cfg.scene.num_envs),
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "max_iterations": int(args.max_iterations),
        "completed_iterations_at_start": completed_iterations,
        "remaining_iterations": remaining_iterations,
        "resume": args.resume,
        "command": " ".join(sys.argv),
        "agent": agent_cfg.to_dict(),
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(f"[DICE] Output directory : {output_dir}")
    print(f"[DICE] Environments     : {env_cfg.scene.num_envs}")
    print(f"[DICE] Rollout length   : {agent_cfg.num_steps_per_env}")
    print(f"[DICE] Target iteration : {args.max_iterations}")
    print(f"[DICE] Remaining        : {remaining_iterations}")

    interrupted = False
    step_chunk = 10

    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=args.max_iterations,
            initial=completed_iterations,
            desc="[DICE RSL-RL Training]",
            unit="iter",
            dynamic_ncols=True,
        )
    except ImportError:
        pbar = None

    try:
        while completed_iterations < args.max_iterations:
            chunk = min(step_chunk, args.max_iterations - completed_iterations)
            runner.learn(
                num_learning_iterations=chunk,
                init_at_random_ep_len=(completed_iterations == 0 and not bool(args.resume)),
            )
            completed_iterations += chunk

            log = raw_env.unwrapped.extras.get("log", {})
            metrics = {
                "Cmds/Ep": f"{log.get('DICE/commands_in_active_episode', 0.0):.2f}",
                "Align": f"{log.get('DICE/alignment', 0.0):.2f}",
                "DropRate": f"{log.get('DICE/drop_rate_per_step', 0.0):.3f}",
                "Hold": f"{log.get('DICE/hold_progress', 0.0):.2f}",
                "AngVel": f"{log.get('DICE/angular_speed', 0.0):.2f}",
            }
            if pbar is not None:
                pbar.update(chunk)
                pbar.set_postfix(metrics)
            else:
                print(f"[DICE Iter {completed_iterations}/{args.max_iterations}] {metrics}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[DICE] Training interrupted by user. Saving current checkpoint.")
    finally:
        if pbar is not None:
            pbar.close()
        final_checkpoint = output_dir / "model_final.pt"
        runner.save(str(final_checkpoint))
        env.close()

    status = "interrupted" if interrupted else "complete"
    print(f"[DICE] Training {status}. Checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
