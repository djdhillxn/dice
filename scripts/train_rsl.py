"""Train DiceDial with RSL-RL on-GPU PPO + Automatic Curriculum Learning.

This is the primary training entry point.  It replaces the old 3-stage
hard curriculum.  A single ``DiceDial-Shadow-Sequence-v0`` environment runs
for the full duration; the AclCurriculum callback tightens success thresholds
automatically as the policy improves.

Quick start
-----------
::

    python scripts/train_rsl.py \\
        --num_envs 2048 \\
        --max_iterations 50000 \\
        --run_name strong_run \\
        --headless

Resume from a checkpoint
------------------------
::

    python scripts/train_rsl.py \\
        --num_envs 2048 \\
        --max_iterations 50000 \\
        --resume outputs/DiceDial-Shadow-Sequence-v0/strong_run/model_*.pt \\
        --run_name strong_run_continued \\
        --headless

Checkpoint format
-----------------
RSL-RL saves ``model_<iter>.pt`` files containing the actor, critic, and
optimiser state.  The ACL state is saved alongside in
``acl_state_<iter>.json``.
"""

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


# ------------------------------------------------------------------ arg parse
parser = argparse.ArgumentParser(
    description="Train DiceDial with RSL-RL PPO + ACL.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--task",
    default="DiceDial-Shadow-Sequence-v0",
    help="Gymnasium environment ID.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=2048,
    help="Number of parallel simulation environments.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=50_000,
    help="Total RSL-RL training iterations (each = one rollout + update).",
)
parser.add_argument(
    "--run_name",
    default=None,
    help="Human-readable tag appended to the output directory name.",
)
parser.add_argument(
    "--output_root",
    default="outputs",
    help="Root directory for training outputs.",
)
parser.add_argument(
    "--resume",
    default=None,
    help="Path to an RSL-RL .pt checkpoint to resume from.",
)
parser.add_argument(
    "--acl_window",
    type=int,
    default=50,
    help="Number of ACL-check calls over which to compute the rolling mean.",
)
parser.add_argument(
    "--acl_check_interval",
    type=int,
    default=100,
    help="RSL-RL iterations between each ACL advancement check.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ late imports
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import dicedial.tasks  # noqa: F401, E402  — registers Gymnasium envs

from dicedial.agents.rsl_rl_ppo_cfg import DICEDIAL_RSL_RL_CFG  # noqa: E402
from dicedial.curriculum import AclCurriculum  # noqa: E402

# RSL-RL / Isaac Lab wrappers — try both known import paths.
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import]
except ImportError:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import]

try:
    from rsl_rl.runners import OnPolicyRunner  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "rsl_rl is not installed.  Install it with:\n"
        "  pip install rsl-rl\n"
        "or follow the Isaac Lab RSL-RL setup guide."
    ) from exc


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or timestamp
    output_dir = Path(args.output_root).resolve() / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- env setup
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device or "cuda:0",
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)

    # ---------------------------------------------------------------- runner
    runner_cfg = {k: dict(v) for k, v in DICEDIAL_RSL_RL_CFG.items()}
    runner_cfg["runner"]["run_name"] = run_name
    runner_cfg["runner"]["max_iterations"] = args.max_iterations

    runner = OnPolicyRunner(
        env,
        runner_cfg,
        log_dir=str(output_dir),
        device=args.device or "cuda:0",
    )

    if args.resume:
        runner.load(args.resume)
        print(f"[DiceDial] Resumed from {args.resume}")

    # ---------------------------------------------------------------- ACL
    unwrapped = raw_env.unwrapped
    acl = AclCurriculum(unwrapped, window=args.acl_window)

    # Restore ACL state if resuming.
    if args.resume:
        acl_path = Path(args.resume).with_suffix(".acl.json")
        if acl_path.exists():
            acl.load_state_dict(json.loads(acl_path.read_text()))
            print(f"[DiceDial] ACL state restored from {acl_path}  "
                  f"(level={acl.current_level['name']})")

    # ---------------------------------------------------------------- metadata
    metadata = {
        "task": args.task,
        "num_envs": env_cfg.scene.num_envs,
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "acl_window": args.acl_window,
        "acl_check_interval": args.acl_check_interval,
        "resume": args.resume,
        "command": " ".join(sys.argv),
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))

    print(f"[DiceDial] Output  : {output_dir}")
    print(f"[DiceDial] Envs    : {env_cfg.scene.num_envs}")
    print(f"[DiceDial] Iters   : {args.max_iterations}")
    print(f"[DiceDial] ACL     : window={args.acl_window}, "
          f"check_every={args.acl_check_interval} iters")

    # ---------------------------------------------------------------- training loop
    completed_iters = 0
    chunk_size = args.acl_check_interval

    with contextlib.suppress(KeyboardInterrupt):
        while completed_iters < args.max_iterations:
            this_chunk = min(chunk_size, args.max_iterations - completed_iters)

            runner.learn(
                num_learning_iterations=this_chunk,
                init_at_random_ep_len=(completed_iters == 0),
            )
            completed_iters += this_chunk

            # Check and possibly advance the ACL.
            log = unwrapped.extras.get("log", {})
            advanced = acl.step(log, runner.it)

            if advanced:
                # Save the model when the curriculum advances so we can
                # inspect performance at each level independently.
                level_name = acl.current_level["name"].replace(" ", "_")
                ckpt_path = output_dir / f"model_acl_{level_name}.pt"
                runner.save(str(ckpt_path))
                acl_state_path = ckpt_path.with_suffix(".acl.json")
                acl_state_path.write_text(json.dumps(acl.state_dict(), indent=2))
                print(f"[DiceDial] ACL checkpoint saved: {ckpt_path}")

    # ---------------------------------------------------------------- final save
    final_model = output_dir / "model_final.pt"
    runner.save(str(final_model))
    acl_final = output_dir / "model_final.acl.json"
    acl_final.write_text(json.dumps(acl.state_dict(), indent=2))

    env.close()
    print(f"[DiceDial] Training complete.  Model: {final_model}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
