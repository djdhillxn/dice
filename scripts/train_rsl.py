"""Train DICE once on the complete final task with RSL-RL PPO."""

import argparse
import faulthandler
import importlib.metadata as package_metadata
import json
import sys
from datetime import datetime
from pathlib import Path


EXPECTED_NUMPY = "1.26.0"


def require_compatible_numpy():
    """Fail before Isaac Sim starts if pip has replaced Isaac Sim's NumPy."""

    try:
        numpy_version = package_metadata.version("numpy")
    except package_metadata.PackageNotFoundError:
        raise SystemExit(
            "[DICE] NumPy is not installed. Activate the 'dice' Conda environment "
            "and reinstall the project before launching Isaac Sim."
        )

    if numpy_version != EXPECTED_NUMPY:
        raise SystemExit(
            "[DICE] Refusing to launch Isaac Sim with NumPy "
            f"{numpy_version}. This DICE/Isaac Sim 5.1 environment requires "
            f"numpy=={EXPECTED_NUMPY}. Repair it with:\n\n"
            "  python -m pip uninstall -y opencv-python opencv-contrib-python "
            "opencv-python-headless opencv-contrib-python-headless\n"
            "  python -m pip install --upgrade 'numpy==1.26.0' "
            "'opencv-python-headless==4.11.0.86'\n"
            "  python -m pip install -e '.[video]'\n"
        )


require_compatible_numpy()

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


import hashlib
import subprocess


def write_metadata(path, metadata):
    path.write_text(json.dumps(metadata, indent=2, default=str))


def get_git_metadata():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8")
        is_dirty = bool(diff.strip())
        return {"commit": commit, "is_dirty": is_dirty, "diff": diff}
    except Exception:
        return {"commit": "unknown", "is_dirty": False, "diff": ""}


def get_runtime_system_metadata(device):
    versions = {}
    for pkg in ("numpy", "torch", "rsl-rl", "rsl-rl-lib", "isaaclab"):
        try:
            versions[pkg] = package_metadata.version(pkg)
        except package_metadata.PackageNotFoundError:
            versions[pkg] = "not_found"

    cuda_version = getattr(torch.version, "cuda", "unknown") if "torch" in sys.modules else "unknown"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    return {
        "packages": versions,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
    }


def get_reward_scale_hash(env_cfg):
    reward_scales = {}
    for attr in dir(env_cfg):
        if "scale" in attr or "bonus" in attr or "penalty" in attr or "reward" in attr:
            val = getattr(env_cfg, attr, None)
            if isinstance(val, (int, float)):
                reward_scales[attr] = val
    serialized = json.dumps(reward_scales, sort_keys=True)
    scale_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return scale_hash, reward_scales


def startup_log(output_dir, message):
    """Print and persist startup milestones so a launch stall has a location."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [DICE STARTUP] {message}"
    print(line, flush=True)
    with (output_dir / "startup.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


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

    total_transitions = int(env_cfg.scene.num_envs) * int(agent_cfg.num_steps_per_env) * int(args.max_iterations)
    reward_hash, reward_scales = get_reward_scale_hash(env_cfg)

    metadata_path = output_dir / "run.json"
    metadata = {
        "project": "DICE",
        "status": "initializing",
        "task": args.task,
        "seed": args.seed,
        "device": device,
        "num_envs": int(env_cfg.scene.num_envs),
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "max_iterations": int(args.max_iterations),
        "total_transitions": total_transitions,
        "observation_space": int(getattr(env_cfg, "observation_space", 121)),
        "action_space": int(getattr(env_cfg, "action_space", 20)),
        "resume": args.resume,
        "command": " ".join(sys.argv),
        "git": get_git_metadata(),
        "runtime_system": get_runtime_system_metadata(device),
        "reward_scale_hash": reward_hash,
        "reward_scales": reward_scales,
        "agent": agent_cfg.to_dict(),
    }
    write_metadata(metadata_path, metadata)

    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        startup_log(output_dir, f"Output directory: {output_dir}")
        startup_log(output_dir, f"Creating Gym environment '{args.task}' with {env_cfg.scene.num_envs} envs...")
        raw_env = gym.make(args.task, cfg=env_cfg)
        startup_log(output_dir, "Gym environment created.")

        startup_log(output_dir, "Creating RSL-RL wrapper (this performs the initial environment reset)...")
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        startup_log(output_dir, f"RSL-RL wrapper/reset complete (obs dim: {env.num_obs}, act dim: {env.num_actions}).")

        startup_log(output_dir, "Constructing OnPolicyRunner and PPO storage...")
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=str(output_dir),
            device=agent_cfg.device,
        )
        startup_log(output_dir, "OnPolicyRunner constructed.")
    finally:
        faulthandler.cancel_dump_traceback_later()

    # RSL-RL already records its own repository diff. Add the DICE repository
    # as well so every run captures the exact local code state used for training.
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)

    resumed = False
    if args.resume:
        checkpoint = compatible_checkpoint_path(Path(args.resume).expanduser().resolve())
        runner.load(checkpoint)
        # RSL-RL checkpoints store the last completed zero-based iteration. Start
        # from the following iteration rather than repeating the checkpoint step.
        runner.current_learning_iteration = int(runner.current_learning_iteration) + 1
        resumed = True
        startup_log(output_dir, f"Resumed from: {checkpoint}")

    start_iteration = int(getattr(runner, "current_learning_iteration", 0))
    remaining_iterations = max(args.max_iterations - start_iteration, 0)

    metadata.update(
        {
            "status": "runner_ready",
            "start_iteration": start_iteration,
            "remaining_iterations": remaining_iterations,
        }
    )
    write_metadata(metadata_path, metadata)

    print(f"[DICE] Output directory : {output_dir}", flush=True)
    print(f"[DICE] Environments     : {env_cfg.scene.num_envs}", flush=True)
    print(f"[DICE] Rollout length   : {agent_cfg.num_steps_per_env}", flush=True)
    print(f"[DICE] Target iteration : {args.max_iterations}", flush=True)
    print(f"[DICE] Remaining        : {remaining_iterations}", flush=True)

    interrupted = False
    failed = False
    final_checkpoint = output_dir / "model_final.pt"

    try:
        if remaining_iterations > 0:
            metadata["status"] = "training"
            write_metadata(metadata_path, metadata)
            startup_log(
                output_dir,
                "Starting RSL-RL learning loop. Native RSL-RL metrics will print after every PPO iteration.",
            )
            runner.learn(
                num_learning_iterations=remaining_iterations,
                init_at_random_ep_len=(start_iteration == 0 and not resumed),
            )
        else:
            startup_log(output_dir, "Target iteration already reached; no PPO updates required.")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[DICE] Training interrupted by user. Saving current checkpoint.", flush=True)
    except Exception:
        failed = True
        metadata["status"] = "failed"
        metadata["last_iteration"] = int(getattr(runner, "current_learning_iteration", 0))
        write_metadata(metadata_path, metadata)
        raise
    finally:
        runner.save(str(final_checkpoint))
        env.close()

    status = "interrupted" if interrupted else "failed" if failed else "complete"
    metadata["status"] = status
    metadata["last_iteration"] = int(getattr(runner, "current_learning_iteration", 0))
    metadata["final_checkpoint"] = str(final_checkpoint)
    write_metadata(metadata_path, metadata)
    print(f"[DICE] Training {status}. Checkpoint: {final_checkpoint}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
