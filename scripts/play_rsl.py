"""Render a deterministic six-command DICE demonstration with RSL-RL."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Render a DICE RSL-RL checkpoint.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--task", default="DICE-Shadow-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="videos/DICE")
parser.add_argument("--video_length", type=int, default=2400)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no_video", action="store_true")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.no_video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
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


def _first(extras, key, default=0):
    value = extras.get(key)
    if value is None:
        return default
    if hasattr(value, "detach"):
        return value[0].detach().cpu().item()
    return value[0]


def main():
    device = args.device or "cuda:0"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_cfg = make_runner_cfg(seed=args.seed, device=device)
    env_cfg = parse_env_cfg(
        args.task,
        device=device,
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    env_cfg.emit_step_metrics = True

    base_env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode=None if args.no_video else "rgb_array",
    )

    wrapped_env = base_env
    if not args.no_video:
        wrapped_env = gym.wrappers.RecordVideo(
            wrapped_env,
            video_folder=str(output_dir / "raw"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            name_prefix="DICE",
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(wrapped_env, clip_actions=agent_cfg.clip_actions)
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

    rows = []
    command_count = len(env_cfg.target_sequence)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.video_length, desc="[DICE Playback Rendering]", unit="step", dynamic_ncols=True)
    except ImportError:
        pbar = None

    for step in range(args.video_length):
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, extras = env.step(actions)

        commands_completed = int(_first(extras, "dice_commands_completed", 0))
        target_f = int(_first(extras, "dice_target_face", 0))
        top_f = int(_first(extras, "dice_top_face", 0))
        align_v = float(_first(extras, "dice_alignment", 0.0))

        rows.append(
            {
                "step": step,
                "target_face": target_f,
                "top_face": top_f,
                "alignment": align_v,
                "position_error": float(_first(extras, "dice_position_error", 0.0)),
                "hold_progress": float(_first(extras, "dice_hold_progress", 0.0)),
                "commands_completed": commands_completed,
                "reward": float(rewards[0].detach().cpu().item()),
            }
        )

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "Target": target_f,
                "Top": top_f,
                "Align": f"{align_v:.2f}",
                "Cmds": commands_completed,
            })

        if hasattr(policy, "reset"):
            policy.reset(dones)

        if commands_completed >= command_count or bool(dones[0].item()):
            break

    if pbar is not None:
        pbar.close()

    metrics_path = output_dir / "video_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    env.close()

    print(f"[DICE] Checkpoint   : {checkpoint}")
    print(f"[DICE] Raw video   : {output_dir / 'raw'}")
    print(f"[DICE] Overlay CSV : {metrics_path}")

    if not args.no_video:
        raw_mp4s = sorted((output_dir / "raw").glob("*.mp4"))
        if raw_mp4s:
            raw_video = raw_mp4s[0]
            annotated_video = output_dir / "DICE_annotated.mp4"
            print(f"[DICE] Found raw video: {raw_video.name}")
            print(f"[DICE] Auto-annotating video -> {annotated_video.name}...")
            try:
                import sys
                sys.path.append(str(Path(__file__).resolve().parent))
                from annotate_video import annotate_video
                annotate_video(raw_video, metrics_path, annotated_video)
            except Exception as exc:
                print(f"[DICE] Warning: Auto-annotation failed: {exc}")
        else:
            print("[DICE] Warning: No raw MP4 file found in raw video directory.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
