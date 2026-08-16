"""Build the complete three-video DICE portfolio package."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont

from dicedial.checkpoint_sweep import resolve_run_directory
from dicedial.portfolio_video import (
    EXPORT_FPS,
    PORTFOLIO_SCHEMA_VERSION,
    RAW_FPS,
    compare_metric_traces,
    compare_physics_snapshots,
    parse_resolution,
    parse_seed_spec,
    probe_portfolio_video,
    read_json,
    read_metric_rows,
    select_representative_scout,
    sha256_file,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the DICE hero, task-explainer, and robustness-boundary "
            "portfolio videos from one completed run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run",
        help="Timestamped run ID below outputs/ or a direct run-directory path.",
    )
    parser.add_argument("--outputs-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--checkpoint", default="model_4000.pt")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "videos"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--raw-fps", type=int, default=RAW_FPS)
    parser.add_argument("--export-fps", type=int, default=EXPORT_FPS)
    parser.add_argument("--nominal-seeds", default="7:11")
    parser.add_argument("--robust-seeds", default="17:21")
    parser.add_argument("--adverse-seeds", default="7:22")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Show the Isaac Sim UI instead of rendering headlessly on the VM.",
    )
    parser.add_argument(
        "--skip-physics-audit",
        action="store_true",
        help="Skip the stock-versus-numbered-die mass/inertia audit.",
    )
    parser.add_argument(
        "--webm",
        action="store_true",
        help="Also create VP9 WebM versions of the three final MP4 files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this run/checkpoint's existing portfolio output directory.",
    )
    values = parser.parse_args()
    try:
        values.resolution_tuple = parse_resolution(values.resolution)
        values.seed_sets = {
            "nominal": parse_seed_spec(values.nominal_seeds),
            "robust": parse_seed_spec(values.robust_seeds),
            "adverse": parse_seed_spec(values.adverse_seeds),
        }
    except ValueError as exc:
        parser.error(str(exc))
    if values.raw_fps <= 0 or values.export_fps <= 0:
        parser.error("Frame rates must be positive")
    return values


def _resolve_executable(value, label):
    path = shutil.which(value)
    if path:
        return path
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(f"{label} executable not found: {value}")


def _verify_video_toolchain(ffmpeg, webm):
    missing_modules = [
        name for name in ("cv2", "moviepy") if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        raise ModuleNotFoundError(
            "Missing video Python modules "
            f'{missing_modules}; run `python -m pip install -e ".[video]"`.'
        )

    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    encoders = result.stdout + result.stderr
    required = ["libx264", "libwebp"]
    if webm:
        required.append("libvpx-vp9")
    missing_encoders = [encoder for encoder in required if encoder not in encoders]
    if missing_encoders:
        raise RuntimeError(
            f"FFmpeg is missing required encoders: {', '.join(missing_encoders)}"
        )


def _run(command):
    print("[DICE PORTFOLIO] $ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _git_metadata():
    def command(*parts):
        result = subprocess.run(
            ["git", *parts],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--short"),
    }


def _installed_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _executable_version(path):
    result = subprocess.run(
        [path, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0]


def _play_command(
    args,
    checkpoint,
    condition,
    seed,
    output,
    camera="hero",
    no_video=False,
    trajectory_input=None,
    trajectory_output=None,
    task=None,
    video_length=None,
):
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts" / "play_rsl.py"),
        "--checkpoint",
        str(checkpoint),
        "--condition",
        condition,
        "--seed",
        str(seed),
        "--camera",
        camera,
        "--resolution",
        args.resolution,
        "--fps",
        str(args.raw_fps),
        "--device",
        args.device,
        "--output",
        str(output),
    ]
    if task:
        command.extend(["--task", task])
    if video_length is not None:
        command.extend(["--video_length", str(video_length)])
    if trajectory_input:
        command.extend(["--trajectory-input", str(trajectory_input)])
    if trajectory_output:
        command.extend(["--trajectory-output", str(trajectory_output)])
    if no_video:
        command.append("--no_video")
    else:
        command.append("--enable_cameras")
        command.append("--kit_args=--/rtx/post/dlss/execMode=2")
    if not args.show_ui:
        command.append("--headless")
    return command


def _run_play(args, checkpoint, condition, seed, output, **kwargs):
    output = Path(output)
    _run(
        _play_command(
            args,
            checkpoint,
            condition,
            seed,
            output,
            **kwargs,
        )
    )
    summary = read_json(output / "capture_summary.json")
    if summary.get("status") != "complete":
        raise RuntimeError(f"Capture did not complete: {output}")
    if summary.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"Capture used the wrong checkpoint: {output}")
    if summary.get("condition") != condition or int(summary.get("seed")) != int(seed):
        raise RuntimeError(f"Capture condition/seed contract failed: {output}")
    return summary


def _run_annotation(args, ffmpeg, capture, output, style, post_roll=0.75):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "annotate_video.py"),
        "--video",
        capture["raw_video"],
        "--metrics",
        capture["metrics"],
        "--summary",
        str(Path(capture["metrics"]).parent / "capture_summary.json"),
        "--initial-metrics",
        capture["initial_metrics"],
        "--output",
        str(output),
        "--style",
        style,
        "--output-fps",
        str(args.export_fps),
        "--crf",
        "16",
        "--post-roll-seconds",
        str(post_roll),
        "--ffmpeg",
        ffmpeg,
    ]
    _run(command)
    return output


def _font(size, bold=False):
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _create_card(path, resolution, eyebrow, title, lines, accent=(70, 211, 124)):
    width, height = resolution
    image = Image.new("RGB", resolution, (11, 16, 24))
    draw = ImageDraw.Draw(image)
    margin = round(width * 0.09)
    draw.rounded_rectangle(
        (margin, round(height * 0.20), width - margin, round(height * 0.80)),
        radius=28,
        fill=(18, 25, 36),
        outline=(55, 68, 86),
        width=2,
    )
    draw.text(
        (margin + 64, round(height * 0.27)),
        eyebrow.upper(),
        font=_font(32, True),
        fill=accent,
    )
    draw.text(
        (margin + 64, round(height * 0.35)),
        title,
        font=_font(64, True),
        fill=(246, 248, 251),
    )
    y = round(height * 0.49)
    for line in lines:
        draw.text(
            (margin + 64, y),
            line,
            font=_font(34, False),
            fill=(191, 202, 216),
        )
        y += 54
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _compose_sequence(ffmpeg, elements, output, resolution, fps, crf=20):
    width, height = resolution
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    filters = []
    labels = []
    for index, element in enumerate(elements):
        if element["type"] == "image":
            command.extend(
                [
                    "-loop",
                    "1",
                    "-framerate",
                    str(fps),
                    "-t",
                    str(element["duration"]),
                    "-i",
                    str(element["path"]),
                ]
            )
            source_filter = (
                f"[{index}:v]trim=duration={element['duration']},setpts=PTS-STARTPTS"
            )
        else:
            command.extend(["-i", str(element["path"])])
            start = float(element.get("start", 0.0))
            end = element.get("end")
            trim = f"trim=start={start}"
            if end is not None:
                trim += f":end={float(end)}"
            speed = float(element.get("speed", 1.0))
            source_filter = f"[{index}:v]{trim},setpts=(PTS-STARTPTS)/{speed}"
        label = f"v{index}"
        source_filter += (
            f",scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=#0b1018,"
            f"fps={fps},setsar=1,format=yuv420p[{label}]"
        )
        filters.append(source_filter)
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[outv]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run(command)
    return output


def _side_by_side(ffmpeg, left, right, output, duration, resolution, fps):
    width, height = resolution
    half_width = width // 2
    panel_height = round(height * 0.56)
    pad_y = (height - panel_height) // 2
    filter_graph = (
        f"[0:v]trim=duration={duration},setpts=PTS-STARTPTS,"
        f"scale={half_width}:{panel_height},fps={fps},setsar=1[left];"
        f"[1:v]trim=duration={duration},setpts=PTS-STARTPTS,"
        f"scale={half_width}:{panel_height},fps={fps},setsar=1[right];"
        f"[left][right]hstack=inputs=2,"
        f"pad={width}:{height}:0:{pad_y}:color=#0b1018,format=yuv420p[outv]"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output


def _extract_image(ffmpeg, video, output, time_seconds, width=1280, webp=False):
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, time_seconds)),
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
    ]
    if webp:
        command.extend(["-c:v", "libwebp", "-quality", "82"])
    command.append(str(output))
    _run(command)
    return output


def _create_contact_sheet(images, labels, output):
    opened = [Image.open(path).convert("RGB") for path in images]
    panel_width, panel_height = 640, 360
    canvas = Image.new(
        "RGB", (panel_width * len(opened), panel_height + 70), (11, 16, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(opened, labels)):
        image.thumbnail((panel_width, panel_height))
        x = index * panel_width + (panel_width - image.width) // 2
        y = 70 + (panel_height - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text(
            (index * panel_width + 24, 20),
            label,
            font=_font(28, True),
            fill=(236, 240, 246),
        )
    canvas.save(output, quality=86)
    return output


def _transcode_webm(ffmpeg, source, output):
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "31",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def _load_final_results(run_directory, checkpoint):
    final_summary = (
        run_directory / "evaluation" / f"final_{checkpoint.stem}" / "final_summary.json"
    )
    if not final_summary.is_file():
        raise FileNotFoundError(
            "The matching quantitative final evaluation is required before "
            f"portfolio rendering: {final_summary}"
        )

    payload = read_json(final_summary)
    expected_hash = sha256_file(checkpoint)
    if payload.get("status") != "complete":
        raise ValueError(f"Final evaluation is not complete: {final_summary}")
    if payload.get("checkpoint_sha256") != expected_hash:
        raise ValueError(
            "Final evaluation checkpoint hash does not match the rendered "
            f"checkpoint: {final_summary}"
        )
    rows = {row["condition"]: row for row in payload["comparison"]}
    missing = {"nominal", "robust", "adverse"} - rows.keys()
    if missing:
        raise ValueError(
            f"Final evaluation is missing conditions {sorted(missing)}: {final_summary}"
        )
    return rows, final_summary


def _adverse_selection_target(final_summary):
    evaluation_directory = final_summary.parent
    adverse_summary_path = evaluation_directory / "adverse" / "summary.json"
    adverse_episodes_path = evaluation_directory / "adverse" / "episodes.csv"
    if not adverse_summary_path.is_file() or not adverse_episodes_path.is_file():
        raise FileNotFoundError(
            "Adverse per-episode artifacts are required for representative "
            f"selection below {evaluation_directory}"
        )
    adverse_summary = read_json(adverse_summary_path)
    step_dt = float(adverse_summary["control_dt_seconds"])
    with adverse_episodes_path.open("r", encoding="utf-8", newline="") as stream:
        dropped_rows = [
            row
            for row in csv.DictReader(stream)
            if row["dropped"].strip().lower() in {"1", "true"}
        ]
    if not dropped_rows:
        raise ValueError(
            f"No dropped episodes were recorded in {adverse_episodes_path}"
        )
    return {
        "commands_completed": median(
            float(row["commands_completed"]) for row in dropped_rows
        ),
        "duration_seconds": median(
            float(row["episode_length"]) * step_dt for row in dropped_rows
        ),
        "dropped_episode_count": len(dropped_rows),
        "source": str(adverse_episodes_path),
    }


def _result_value(row, key):
    return float(row[key])


def _success_times(metrics_path):
    return [
        float(row["sim_time_seconds"])
        for row in read_metric_rows(metrics_path)
        if int(float(row["success"])) == 1
    ]


def main():
    args = parse_args()
    ffmpeg = _resolve_executable(args.ffmpeg, "ffmpeg")
    ffprobe = _resolve_executable(args.ffprobe, "ffprobe")
    _verify_video_toolchain(ffmpeg, args.webm)
    run_directory = resolve_run_directory(args.run, args.outputs_root)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = run_directory / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    final_results, final_results_source = _load_final_results(run_directory, checkpoint)
    adverse_selection_target = _adverse_selection_target(final_results_source)

    output_root = Path(args.output_root).expanduser().resolve()
    output_directory = output_root / f"{run_directory.name}_{checkpoint.stem}"
    expected_name = f"{run_directory.name}_{checkpoint.stem}"
    if output_directory.exists():
        if not args.force:
            raise FileExistsError(
                f"Portfolio output already exists: {output_directory}. Pass --force to replace it."
            )
        if (
            output_directory.parent != output_root
            or output_directory.name != expected_name
        ):
            raise RuntimeError("Refusing to remove an unexpected portfolio output path")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)

    state_path = output_directory / "manifest.json"
    state = {
        "schema_version": PORTFOLIO_SCHEMA_VERSION,
        "project": "DICE",
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "run_directory": str(run_directory),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "final_results_source": str(final_results_source),
        "final_results_source_sha256": sha256_file(final_results_source),
        "resolution": list(args.resolution_tuple),
        "raw_fps": args.raw_fps,
        "export_fps": args.export_fps,
        "seed_candidates": args.seed_sets,
        "adverse_selection_target": adverse_selection_target,
        "git": _git_metadata(),
        "software": {
            "python": sys.version,
            "isaaclab": _installed_version("isaaclab"),
            "isaacsim": _installed_version("isaacsim"),
            "rsl_rl": _installed_version("rsl-rl-lib", "rsl-rl"),
            "torch": _installed_version("torch"),
            "opencv": _installed_version("opencv-python-headless"),
            "pillow": _installed_version("Pillow"),
            "moviepy": _installed_version("moviepy"),
            "ffmpeg": _executable_version(ffmpeg),
            "ffprobe": _executable_version(ffprobe),
        },
        "phases": [],
    }
    write_json(state_path, state)

    try:
        audit = None
        if not args.skip_physics_audit:
            audit_root = output_directory / "audit"
            stock = _run_play(
                args,
                checkpoint,
                "nominal",
                7,
                audit_root / "stock",
                no_video=True,
                task="DICE-Shadow-Eval-v0",
                video_length=1,
            )
            presentation = _run_play(
                args,
                checkpoint,
                "nominal",
                7,
                audit_root / "numbered",
                no_video=True,
                video_length=1,
            )
            physics_snapshots = {
                "stock": stock["physics_snapshot"],
                "numbered": presentation["physics_snapshot"],
            }
            write_json(audit_root / "physics_snapshots.json", physics_snapshots)
            audit = compare_physics_snapshots(
                stock["physics_snapshot"], presentation["physics_snapshot"]
            )
            audit.update(physics_snapshots)
            write_json(audit_root / "physics_audit.json", audit)
        state["physics_audit"] = audit or {"status": "skipped"}
        state["phases"].append("physics_audit")
        write_json(state_path, state)

        scout_summaries = {}
        selections = {}
        for condition, seeds in args.seed_sets.items():
            condition_summaries = []
            for seed in seeds:
                scout_directory = (
                    output_directory / "scout" / condition / f"seed_{seed}"
                )
                summary = _run_play(
                    args,
                    checkpoint,
                    condition,
                    seed,
                    scout_directory,
                    no_video=True,
                    trajectory_output=scout_directory / "trajectory.npz",
                )
                condition_summaries.append(summary)
            scout_summaries[condition] = condition_summaries
            selection_kwargs = {}
            if condition == "adverse":
                selection_kwargs = {
                    "adverse_target_commands": adverse_selection_target[
                        "commands_completed"
                    ],
                    "adverse_target_duration_seconds": adverse_selection_target[
                        "duration_seconds"
                    ],
                }
            selections[condition] = select_representative_scout(
                condition,
                condition_summaries,
                **selection_kwargs,
            )
        state["scout_summaries"] = scout_summaries
        state["selections"] = selections
        state["phases"].append("representative_selection")
        write_json(state_path, state)

        capture_plan = {
            "nominal": ("hero", "top", "side"),
            "robust": ("hero",),
            "adverse": ("hero", "side"),
        }
        captures = {}
        trace_checks = {}
        for condition, cameras in capture_plan.items():
            selected = selections[condition]
            seed = int(selected["seed"])
            for camera in cameras:
                capture_directory = (
                    output_directory / "captures" / condition / f"seed_{seed}" / camera
                )
                summary = _run_play(
                    args,
                    checkpoint,
                    condition,
                    seed,
                    capture_directory,
                    camera=camera,
                    trajectory_input=selected["trajectory"],
                )
                key = f"{condition}_{camera}"
                captures[key] = summary
                trace_checks[key] = compare_metric_traces(
                    selected["metrics"], summary["metrics"]
                )
        state["captures"] = captures
        state["trace_checks"] = trace_checks
        state["phases"].append("camera_capture")
        write_json(state_path, state)

        intermediate = output_directory / "intermediate"
        intermediate.mkdir()
        annotations = {
            "nominal_hero_minimal": _run_annotation(
                args,
                ffmpeg,
                captures["nominal_hero"],
                intermediate / "nominal_hero_minimal.mp4",
                "minimal",
                0.75,
            ),
            "nominal_hero_technical": _run_annotation(
                args,
                ffmpeg,
                captures["nominal_hero"],
                intermediate / "nominal_hero_technical.mp4",
                "technical",
                0.35,
            ),
            "nominal_top_technical": _run_annotation(
                args,
                ffmpeg,
                captures["nominal_top"],
                intermediate / "nominal_top_technical.mp4",
                "technical",
                0.35,
            ),
            "nominal_side_technical": _run_annotation(
                args,
                ffmpeg,
                captures["nominal_side"],
                intermediate / "nominal_side_technical.mp4",
                "technical",
                0.35,
            ),
            "robust_hero": _run_annotation(
                args,
                ffmpeg,
                captures["robust_hero"],
                intermediate / "robust_hero.mp4",
                "stress",
                0.35,
            ),
            "adverse_hero": _run_annotation(
                args,
                ffmpeg,
                captures["adverse_hero"],
                intermediate / "adverse_hero.mp4",
                "stress",
                0.75,
            ),
            "adverse_side": _run_annotation(
                args,
                ffmpeg,
                captures["adverse_side"],
                intermediate / "adverse_side.mp4",
                "stress",
                0.75,
            ),
        }
        state["annotations"] = {key: str(value) for key, value in annotations.items()}
        state["phases"].append("annotation")
        write_json(state_path, state)

        nominal_result = final_results["nominal"]
        robust_result = final_results["robust"]
        adverse_result = final_results["adverse"]
        nominal_success = 100.0 * _result_value(
            nominal_result, "issued_command_completion_rate"
        )
        nominal_mean_commands = _result_value(
            nominal_result, "mean_consecutive_commands"
        )
        nominal_episodes = int(_result_value(nominal_result, "episodes"))
        robust_episodes = int(_result_value(robust_result, "episodes"))
        adverse_episodes = int(_result_value(adverse_result, "episodes"))
        episode_counts = {nominal_episodes, robust_episodes, adverse_episodes}
        evaluation_eyebrow = (
            f"{nominal_episodes:,} EPISODES PER CONDITION"
            if len(episode_counts) == 1
            else "FINAL QUANTITATIVE EVALUATION"
        )
        adverse_drops = round(
            adverse_episodes * _result_value(adverse_result, "drop_rate")
        )
        cards = output_directory / "cards"
        hero_card = _create_card(
            cards / "hero.png",
            args.resolution_tuple,
            "DICE",
            "Command-conditioned in-hand reorientation",
            ["20-DoF Shadow Hand", "One policy  |  six semantic face commands"],
        )
        explainer_card = _create_card(
            cards / "explainer.png",
            args.resolution_tuple,
            "TASK",
            "Place the requested numbered face upward",
            ["Rotate  |  stabilize  |  hold all gates for 20 control steps"],
            accent=(68, 196, 224),
        )
        explainer_outro = _create_card(
            cards / "explainer_outro.png",
            args.resolution_tuple,
            "FINAL NOMINAL RESULT",
            f"{nominal_success:.2f}% issued-command completion",
            [
                f"{nominal_mean_commands:.3f} mean commands / episode",
                f"{100.0 * _result_value(nominal_result, 'drop_rate'):.2f}% "
                f"episode drop rate  |  n = {nominal_episodes:,}",
            ],
        )
        hold_replay_card = _create_card(
            cards / "hold_replay.png",
            args.resolution_tuple,
            "0.5x REPLAY",
            "Alignment and hold confirmation",
            ["The command changes only after 20 consecutive valid steps"],
            accent=(68, 196, 224),
        )

        nominal_drop = 100.0 * _result_value(nominal_result, "drop_rate")
        robust_drop = 100.0 * _result_value(robust_result, "drop_rate")
        adverse_drop = 100.0 * _result_value(adverse_result, "drop_rate")
        stress_card = _create_card(
            cards / "stress.png",
            args.resolution_tuple,
            evaluation_eyebrow,
            "Robustness and the failure boundary",
            [
                f"Nominal {nominal_drop:.1f}% drops  |  +/-20% physics {robust_drop:.1f}%",
                f"Adverse 1.5x mass / 0.7 friction {adverse_drop:.1f}% drops",
            ],
            accent=(242, 89, 89),
        )
        slip_replay_card = _create_card(
            cards / "slip_replay.png",
            args.resolution_tuple,
            "0.5x REPLAY",
            "The adverse retention failure",
            ["Heavier object  |  lower tangential contact margin"],
            accent=(242, 89, 89),
        )
        stress_outro = _create_card(
            cards / "stress_outro.png",
            args.resolution_tuple,
            "REPRESENTATIVE ROLLOUT",
            "Semantic targeting remains strong",
            [
                "Long-horizon grasp retention degrades under the adverse corner",
                f"Aggregate result: {adverse_drops:,} / {adverse_episodes:,} "
                "episodes dropped",
            ],
            accent=(242, 89, 89),
        )

        nominal_duration = float(selections["nominal"]["duration_seconds"])
        robust_duration = float(selections["robust"]["duration_seconds"])
        adverse_duration = float(selections["adverse"]["duration_seconds"])
        success_times = _success_times(selections["nominal"]["metrics"])
        if len(success_times) < 6:
            raise RuntimeError(
                "Selected nominal trajectory has fewer than six success events"
            )
        cut_two = success_times[1]
        cut_four = success_times[3]
        replay_end = min(nominal_duration, success_times[4] + 0.20)
        replay_start = max(0.0, replay_end - 1.10)

        comparison = _side_by_side(
            ffmpeg,
            annotations["nominal_hero_technical"],
            annotations["robust_hero"],
            intermediate / "nominal_vs_robust.mp4",
            min(6.0, nominal_duration, robust_duration),
            args.resolution_tuple,
            args.export_fps,
        )

        exports = output_directory / "exports"
        exports.mkdir()
        hero = _compose_sequence(
            ffmpeg,
            [
                {"type": "image", "path": hero_card, "duration": 1.20},
                {"type": "video", "path": annotations["nominal_hero_minimal"]},
            ],
            exports / "dice_hero.mp4",
            args.resolution_tuple,
            args.export_fps,
        )
        explainer = _compose_sequence(
            ffmpeg,
            [
                {"type": "image", "path": explainer_card, "duration": 2.50},
                {
                    "type": "video",
                    "path": annotations["nominal_hero_technical"],
                    "start": 0.0,
                    "end": cut_two,
                },
                {
                    "type": "video",
                    "path": annotations["nominal_top_technical"],
                    "start": cut_two,
                    "end": cut_four,
                },
                {
                    "type": "video",
                    "path": annotations["nominal_side_technical"],
                    "start": cut_four,
                    "end": nominal_duration,
                },
                {"type": "image", "path": hold_replay_card, "duration": 0.80},
                {
                    "type": "video",
                    "path": annotations["nominal_side_technical"],
                    "start": replay_start,
                    "end": replay_end,
                    "speed": 0.5,
                },
                {"type": "image", "path": explainer_outro, "duration": 3.50},
            ],
            exports / "dice_task_explainer.mp4",
            args.resolution_tuple,
            args.export_fps,
        )
        stress = _compose_sequence(
            ffmpeg,
            [
                {"type": "image", "path": stress_card, "duration": 3.50},
                {"type": "video", "path": comparison},
                {
                    "type": "video",
                    "path": annotations["adverse_hero"],
                    "start": 0.0,
                    "end": adverse_duration,
                },
                {"type": "image", "path": slip_replay_card, "duration": 0.80},
                {
                    "type": "video",
                    "path": annotations["adverse_side"],
                    "start": max(0.0, adverse_duration - 2.0),
                    "end": adverse_duration,
                    "speed": 0.5,
                },
                {"type": "image", "path": stress_outro, "duration": 4.00},
            ],
            exports / "dice_robustness_boundary.mp4",
            args.resolution_tuple,
            args.export_fps,
        )

        export_videos = [hero, explainer, stress]
        preferred_poster_times = {
            hero: 2.0,
            explainer: 4.0,
            stress: 5.0,
        }
        posters = []
        for video in export_videos:
            poster = video.with_name(video.stem + "_poster.webp")
            probe = probe_portfolio_video(
                ffprobe,
                video,
                args.resolution_tuple,
                args.export_fps,
            )
            _extract_image(
                ffmpeg,
                video,
                poster,
                min(
                    preferred_poster_times[video],
                    max(0.0, probe["duration_seconds"] - 0.10),
                ),
                webp=True,
            )
            posters.append(poster)

        contact_images = []
        for camera, annotated_key in (
            ("hero", "nominal_hero_technical"),
            ("top", "nominal_top_technical"),
            ("side", "nominal_side_technical"),
        ):
            image_path = intermediate / f"contact_{camera}.png"
            _extract_image(
                ffmpeg,
                annotations[annotated_key],
                image_path,
                nominal_duration * 0.5,
            )
            contact_images.append(image_path)
        contact_sheet = _create_contact_sheet(
            contact_images,
            ["HERO OBLIQUE", "TOP DIAGNOSTIC", "SIDE CONTACT"],
            exports / "camera_contact_sheet.webp",
        )

        webm_outputs = []
        if args.webm:
            for video in export_videos:
                webm = video.with_suffix(".webm")
                _transcode_webm(ffmpeg, video, webm)
                webm_outputs.append(webm)

        export_metadata = [
            probe_portfolio_video(
                ffprobe,
                video,
                args.resolution_tuple,
                args.export_fps,
            )
            for video in export_videos
        ]
        total_export_size_bytes = sum(item["size_bytes"] for item in export_metadata)
        package_size_warning = total_export_size_bytes > 40 * 1024 * 1024
        checksum_paths = [*export_videos, *posters, contact_sheet, *webm_outputs]
        checksums_path = exports / "checksums.sha256"
        checksums_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in sorted(checksum_paths, key=lambda item: item.name)
            ),
            encoding="utf-8",
        )

        state.update(
            {
                "status": "complete",
                "finished_at": datetime.now().astimezone().isoformat(),
                "phases": [
                    *state["phases"],
                    "composition",
                    "poster_generation",
                    "technical_validation",
                ],
                "final_results_source": str(final_results_source),
                "exports": export_metadata,
                "total_mp4_size_bytes": total_export_size_bytes,
                "total_mp4_size_target_bytes": 40 * 1024 * 1024,
                "total_mp4_size_warning": package_size_warning,
                "posters": [str(path) for path in posters],
                "camera_contact_sheet": str(contact_sheet),
                "webm_exports": [str(path) for path in webm_outputs],
                "checksums": str(checksums_path),
            }
        )
        write_json(state_path, state)

        print("\n[DICE PORTFOLIO] Complete", flush=True)
        for metadata in export_metadata:
            print(
                f"  {metadata['path']}  "
                f"{metadata['duration_seconds']:.2f}s  "
                f"{metadata['size_bytes'] / (1024 * 1024):.2f} MiB",
                flush=True,
            )
        print(
            "[DICE PORTFOLIO] Total MP4 payload: "
            f"{total_export_size_bytes / (1024 * 1024):.2f} MiB",
            flush=True,
        )
        if package_size_warning:
            print(
                "[DICE PORTFOLIO] Warning: total MP4 payload exceeds the "
                "40 MiB portfolio target; consider a higher final CRF.",
                flush=True,
            )
        print(f"[DICE PORTFOLIO] Transfer this directory: {exports}", flush=True)
        print(f"[DICE PORTFOLIO] Manifest: {state_path}", flush=True)
    except BaseException as exc:
        state.update(
            {
                "status": "interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
