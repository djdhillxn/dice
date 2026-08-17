"""Build the complete three-video DICE portfolio package."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont, features as pillow_features

from dicedial.checkpoint_sweep import resolve_run_directory
from dicedial.portfolio_video import (
    EXPORT_FPS,
    PORTFOLIO_FINAL_HOLD_SECONDS,
    PORTFOLIO_PLAYBACK_SPEED,
    PORTFOLIO_SCHEMA_VERSION,
    RAW_FPS,
    build_portfolio_captions,
    compare_metric_traces,
    compare_physics_snapshots,
    parse_resolution,
    parse_seed_spec,
    probe_portfolio_video,
    read_json,
    resolve_portfolio_artifact,
    select_representative_scout,
    sha256_file,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture or recompose the three-video DICE portfolio package from "
            "one completed run."
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
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=PORTFOLIO_PLAYBACK_SPEED,
        help="Presentation speed for all policy footage; simulation timing is unchanged.",
    )
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
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help=(
            "Reuse an existing completed capture package and transactionally "
            "replace only its public exports; Isaac Sim is not launched."
        ),
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
    if values.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if values.compose_only and not values.force:
        parser.error("--compose-only requires --force to replace existing exports")
    return values


def _resolve_executable(value, label):
    path = shutil.which(value)
    if path:
        return path
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(f"{label} executable not found: {value}")


def _verify_video_toolchain(ffmpeg, webm, capture_required=True):
    module_names = ["cv2"]
    if capture_required:
        module_names.append("moviepy")
    missing_modules = [
        name for name in module_names if importlib.util.find_spec(name) is None
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
    required = ["libx264"]
    if webm:
        required.append("libvpx-vp9")
    missing_encoders = [encoder for encoder in required if encoder not in encoders]
    if missing_encoders:
        raise RuntimeError(
            f"FFmpeg is missing required encoders: {', '.join(missing_encoders)}"
        )
    if not pillow_features.check("webp"):
        raise RuntimeError("Pillow was built without required WebP support")


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


def _run_annotation(
    args,
    ffmpeg,
    capture,
    output,
    style,
    post_roll=PORTFOLIO_FINAL_HOLD_SECONDS,
    video=None,
    view_labels=(),
    playback_speed=None,
    output_fps=None,
):
    playback_speed = args.playback_speed if playback_speed is None else playback_speed
    output_fps = args.export_fps if output_fps is None else output_fps
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "annotate_video.py"),
        "--video",
        str(video or capture["raw_video"]),
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
        str(output_fps),
        "--crf",
        "16",
        "--post-roll-seconds",
        str(post_roll),
        "--playback-speed",
        str(playback_speed),
        "--ffmpeg",
        ffmpeg,
    ]
    for label in view_labels:
        command.extend(["--view-label", label])
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


def _compose_synchronized_views(ffmpeg, left, right, output, resolution, fps):
    """Crop two synchronized 16:9 captures into readable full-height panels."""

    width, height = resolution
    half_width = width // 2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_filter = (
        f"crop=iw/2:ih:(iw-ow)/2:0,scale={half_width}:{height},fps={fps},setsar=1"
    )
    filter_graph = (
        f"[0:v]{panel_filter}[left];"
        f"[1:v]{panel_filter}[right];"
        "[left][right]hstack=inputs=2:shortest=1,"
        f"drawbox=x={half_width - 1}:y=0:w=2:h={height}:"
        "color=#d8dee8@0.45:t=fill,format=yuv420p[outv]"
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


def _create_comparison_overlay(path, resolution, playback_speed):
    """Create a transparent, temporary legend—not a standalone title card."""

    width, height = resolution
    image = Image.new("RGBA", resolution, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(28, True)
    small_font = _font(23, True)
    margin = 42
    chip_height = 58
    chip_width = 390
    for left, label, accent in (
        (margin, "NOMINAL", (70, 211, 124, 255)),
        (width // 2 + margin, "±20% PHYSICS VARIATION", (68, 196, 224, 255)),
    ):
        draw.rounded_rectangle(
            (left, margin, left + chip_width, margin + chip_height),
            radius=14,
            fill=(15, 20, 28, 215),
            outline=(255, 255, 255, 40),
            width=1,
        )
        draw.text((left + 20, margin + 12), label, font=font, fill=accent)
    speed_text = f"{playback_speed:g}x PLAYBACK"
    speed_box = (
        width - margin - 260,
        height - margin - 54,
        width - margin,
        height - margin,
    )
    draw.rounded_rectangle(speed_box, radius=12, fill=(15, 20, 28, 205))
    draw.text(
        (speed_box[0] + 18, speed_box[1] + 12),
        speed_text,
        font=small_font,
        fill=(226, 232, 240, 255),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _compose_physics_variation(
    ffmpeg,
    nominal_video,
    robust_video,
    overlay,
    output,
    nominal_duration,
    robust_duration,
    resolution,
    output_fps,
    playback_speed,
    final_hold_seconds,
):
    """Compose complete median-like nominal/variation rollouts side by side."""

    width, height = resolution
    half_width = width // 2
    longest_duration = max(nominal_duration, robust_duration)
    nominal_pad = (longest_duration - nominal_duration) / playback_speed
    robust_pad = (longest_duration - robust_duration) / playback_speed

    def panel(index, duration, padding, label):
        return (
            f"[{index}:v]trim=duration={duration},"
            f"setpts=(PTS-STARTPTS)/{playback_speed},"
            f"crop=iw/2:ih:(iw-ow)/2:0,scale={half_width}:{height},"
            f"fps={output_fps},tpad=stop_mode=clone:"
            f"stop_duration={padding + final_hold_seconds},setsar=1[{label}]"
        )

    filter_graph = ";".join(
        (
            panel(0, nominal_duration, nominal_pad, "left"),
            panel(1, robust_duration, robust_pad, "right"),
            "[left][right]hstack=inputs=2:shortest=1[views]",
            "[views][2:v]overlay=0:0:shortest=1,format=yuv420p[outv]",
        )
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(nominal_video),
            "-i",
            str(robust_video),
            "-loop",
            "1",
            "-framerate",
            str(output_fps),
            "-i",
            str(overlay),
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
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return output


def _replace_exports_transactionally(staged_exports, destination):
    """Install validated exports while retaining rollback until replacement succeeds."""

    staged_exports = Path(staged_exports)
    destination = Path(destination)
    backup = destination.parent / f".{destination.name}.previous"
    if backup.exists():
        raise FileExistsError(f"Stale export rollback directory exists: {backup}")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staged_exports, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _extract_image(ffmpeg, video, output, time_seconds, width=1280, webp=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = None
    encoded_output = output
    if webp:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix=".portfolio-poster-",
            dir=output.parent,
        )
        encoded_output = Path(temporary_directory.name) / "frame.png"
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
    command.append(str(encoded_output))
    try:
        _run(command)
        if webp:
            with Image.open(encoded_output) as image:
                image.save(output, format="WEBP", quality=82, method=6)
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
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


def _localized_captures(state, output_directory):
    required = {
        "nominal_hero",
        "nominal_top",
        "robust_hero",
        "adverse_hero",
        "adverse_side",
    }
    recorded = state.get("captures", {})
    missing = required - recorded.keys()
    if missing:
        raise ValueError(f"Portfolio manifest is missing captures: {sorted(missing)}")

    captures = {}
    selections = state.get("selections", {})
    for key in sorted(required):
        expected_condition, expected_camera = key.split("_", maxsplit=1)
        capture = dict(recorded[key])
        for field in ("raw_video", "metrics", "initial_metrics"):
            capture[field] = str(
                resolve_portfolio_artifact(capture[field], output_directory)
            )
        summary_path = Path(capture["metrics"]).parent / "capture_summary.json"
        summary = read_json(summary_path)
        if summary.get("checkpoint_sha256") != state.get("checkpoint_sha256"):
            raise ValueError(f"Capture checkpoint identity mismatch: {summary_path}")
        if (
            summary.get("condition") != expected_condition
            or capture.get("condition") != expected_condition
        ):
            raise ValueError(f"Capture condition mismatch: {summary_path}")
        if (
            summary.get("camera") != expected_camera
            or capture.get("camera") != expected_camera
        ):
            raise ValueError(f"Capture camera mismatch: {summary_path}")
        expected_seed = int(selections[expected_condition]["seed"])
        if int(summary.get("seed")) != expected_seed:
            raise ValueError(f"Capture seed mismatch: {summary_path}")
        if tuple(summary.get("resolution", ())) != tuple(state.get("resolution", ())):
            raise ValueError(f"Capture resolution mismatch: {summary_path}")
        if int(summary.get("raw_fps", 0)) != int(state.get("raw_fps", 0)):
            raise ValueError(f"Capture frame-rate mismatch: {summary_path}")
        captures[key] = capture

    compare_metric_traces(
        captures["nominal_hero"]["metrics"],
        captures["nominal_top"]["metrics"],
    )
    compare_metric_traces(
        captures["adverse_hero"]["metrics"],
        captures["adverse_side"]["metrics"],
    )
    return captures


def _compose_public_package(
    args,
    ffmpeg,
    ffprobe,
    output_directory,
    state,
    final_results,
    composition_mode,
):
    """Build and transactionally install the presentation-only public package."""

    selections = state.get("selections", {})
    if {"nominal", "robust", "adverse"} - selections.keys():
        raise ValueError("Portfolio manifest does not contain all selected rollouts")
    captures = _localized_captures(state, output_directory)

    nominal_duration = float(selections["nominal"]["duration_seconds"])
    robust_duration = float(selections["robust"]["duration_seconds"])
    adverse_duration = float(selections["adverse"]["duration_seconds"])
    final_hold = PORTFOLIO_FINAL_HOLD_SECONDS

    with tempfile.TemporaryDirectory(
        prefix=".portfolio-compose-",
        dir=output_directory,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        intermediate = staging / "intermediate"
        exports = staging / "exports"
        intermediate.mkdir()
        exports.mkdir()

        nominal_views = _compose_synchronized_views(
            ffmpeg,
            captures["nominal_hero"]["raw_video"],
            captures["nominal_top"]["raw_video"],
            intermediate / "nominal_oblique_top.mp4",
            args.resolution_tuple,
            args.raw_fps,
        )
        nominal_annotated = _run_annotation(
            args,
            ffmpeg,
            captures["nominal_hero"],
            intermediate / "dice_nominal_success.mp4",
            "technical",
            final_hold,
            video=nominal_views,
            view_labels=("Oblique", "Top view"),
        )
        nominal_export = exports / "dice_nominal_success.mp4"
        shutil.copy2(nominal_annotated, nominal_export)

        adverse_views = _compose_synchronized_views(
            ffmpeg,
            captures["adverse_hero"]["raw_video"],
            captures["adverse_side"]["raw_video"],
            intermediate / "adverse_oblique_side.mp4",
            args.resolution_tuple,
            args.raw_fps,
        )
        adverse_annotated = _run_annotation(
            args,
            ffmpeg,
            captures["adverse_hero"],
            intermediate / "dice_adverse_boundary.mp4",
            "stress",
            final_hold,
            video=adverse_views,
            view_labels=("Oblique", "Side view"),
        )
        adverse_export = exports / "dice_adverse_boundary.mp4"
        shutil.copy2(adverse_annotated, adverse_export)

        # Normalize the two comparison sources through the strict telemetry
        # aligner. Style "none" leaves pixels untouched while restoring the
        # terminal frame that MoviePy may omit at the encoding boundary.
        nominal_comparison_source = _run_annotation(
            args,
            ffmpeg,
            captures["nominal_hero"],
            intermediate / "nominal_comparison_source.mp4",
            "none",
            0.0,
            playback_speed=1.0,
            output_fps=args.raw_fps,
        )
        robust_comparison_source = _run_annotation(
            args,
            ffmpeg,
            captures["robust_hero"],
            intermediate / "robust_comparison_source.mp4",
            "none",
            0.0,
            playback_speed=1.0,
            output_fps=args.raw_fps,
        )
        comparison_overlay = _create_comparison_overlay(
            intermediate / "physics_variation_overlay.png",
            args.resolution_tuple,
            args.playback_speed,
        )
        variation_export = _compose_physics_variation(
            ffmpeg,
            nominal_comparison_source,
            robust_comparison_source,
            comparison_overlay,
            exports / "dice_physics_variation.mp4",
            nominal_duration,
            robust_duration,
            args.resolution_tuple,
            args.export_fps,
            args.playback_speed,
            final_hold,
        )

        videos = [nominal_export, variation_export, adverse_export]
        expected_durations = {
            nominal_export.name: nominal_duration / args.playback_speed + final_hold,
            variation_export.name: max(nominal_duration, robust_duration)
            / args.playback_speed
            + final_hold,
            adverse_export.name: adverse_duration / args.playback_speed + final_hold,
        }
        poster_fractions = {
            nominal_export.name: 0.50,
            variation_export.name: 0.50,
            adverse_export.name: 0.68,
        }
        export_metadata = []
        posters = []
        for video in videos:
            metadata = probe_portfolio_video(
                ffprobe,
                video,
                args.resolution_tuple,
                args.export_fps,
            )
            expected = expected_durations[video.name]
            tolerance = max(0.12, 3.0 / args.export_fps)
            if abs(metadata["duration_seconds"] - expected) > tolerance:
                raise ValueError(
                    f"Unexpected edited duration for {video.name}: "
                    f"{metadata['duration_seconds']:.3f}s != {expected:.3f}s "
                    f"(tolerance {tolerance:.3f}s)"
                )
            metadata.update(
                {
                    "playback_speed": args.playback_speed,
                    "expected_duration_seconds": expected,
                }
            )
            export_metadata.append(metadata)
            poster = exports / f"{video.stem}_poster.webp"
            _extract_image(
                ffmpeg,
                video,
                poster,
                metadata["duration_seconds"] * poster_fractions[video.name],
                webp=True,
            )
            posters.append(poster)

        captions = build_portfolio_captions(
            final_results,
            selections,
            playback_speed=args.playback_speed,
        )
        caption_paths = []
        for filename, markdown in captions.items():
            path = exports / filename
            path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            caption_paths.append(path)

        webm_outputs = []
        if args.webm:
            for video in videos:
                webm = exports / f"{video.stem}.webm"
                _transcode_webm(ffmpeg, video, webm)
                webm_outputs.append(webm)

        checksum_members = [*videos, *posters, *caption_paths, *webm_outputs]
        checksums_path = exports / "checksums.sha256"
        checksums_path.write_text(
            "".join(
                f"{sha256_file(path)}  {path.name}\n"
                for path in sorted(checksum_members, key=lambda item: item.name)
            ),
            encoding="utf-8",
        )

        destination = output_directory / "exports"
        destination_metadata = []
        for metadata in export_metadata:
            item = dict(metadata)
            item["path"] = str(destination / Path(metadata["path"]).name)
            item["poster"] = str(
                destination / f"{Path(metadata['path']).stem}_poster.webp"
            )
            item["caption"] = str(destination / f"{Path(metadata['path']).stem}.md")
            destination_metadata.append(item)

        _replace_exports_transactionally(exports, destination)

    total_export_size_bytes = sum(item["size_bytes"] for item in destination_metadata)
    state.update(
        {
            "status": "complete",
            "finished_at": datetime.now().astimezone().isoformat(),
            "presentation_revision": 2,
            "composition_mode": composition_mode,
            "playback_speed": args.playback_speed,
            "final_hold_seconds": final_hold,
            "phases": [
                *[
                    phase
                    for phase in state.get("phases", [])
                    if phase
                    not in {
                        "annotation",
                        "composition",
                        "poster_generation",
                        "technical_validation",
                        "markdown_captions",
                    }
                ],
                "composition",
                "poster_generation",
                "markdown_captions",
                "technical_validation",
            ],
            "exports": destination_metadata,
            "total_mp4_size_bytes": total_export_size_bytes,
            "total_mp4_size_target_bytes": 40 * 1024 * 1024,
            "total_mp4_size_warning": total_export_size_bytes > 40 * 1024 * 1024,
            "posters": [
                str(output_directory / "exports" / path.name) for path in posters
            ],
            "captions": [
                str(output_directory / "exports" / path.name) for path in caption_paths
            ],
            "webm_exports": [
                str(output_directory / "exports" / path.name) for path in webm_outputs
            ],
            "checksums": str(output_directory / "exports" / checksums_path.name),
            "composition_git": _git_metadata(),
            "composition_software": {
                "python": sys.version,
                "opencv": _installed_version("opencv-python-headless", "opencv-python"),
                "pillow": _installed_version("Pillow"),
                "ffmpeg": _executable_version(ffmpeg),
                "ffprobe": _executable_version(ffprobe),
            },
        }
    )
    state.pop("annotations", None)
    state.pop("camera_contact_sheet", None)
    write_json(output_directory / "manifest.json", state)

    print("\n[DICE PORTFOLIO] Presentation revision 2 complete", flush=True)
    for metadata in destination_metadata:
        print(
            f"  {metadata['path']}  {metadata['duration_seconds']:.2f}s  "
            f"{metadata['size_bytes'] / (1024 * 1024):.2f} MiB",
            flush=True,
        )
    print(
        f"[DICE PORTFOLIO] Playback: {args.playback_speed:g}× at {args.export_fps} FPS",
        flush=True,
    )
    print(f"[DICE PORTFOLIO] Transfer this directory: {output_directory / 'exports'}")
    return state


def main():
    args = parse_args()
    ffmpeg = _resolve_executable(args.ffmpeg, "ffmpeg")
    ffprobe = _resolve_executable(args.ffprobe, "ffprobe")
    _verify_video_toolchain(
        ffmpeg,
        args.webm,
        capture_required=not args.compose_only,
    )
    run_directory = resolve_run_directory(args.run, args.outputs_root)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = run_directory / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    final_results, final_results_source = _load_final_results(run_directory, checkpoint)

    output_root = Path(args.output_root).expanduser().resolve()
    output_directory = output_root / f"{run_directory.name}_{checkpoint.stem}"
    expected_name = f"{run_directory.name}_{checkpoint.stem}"
    if args.compose_only:
        state_path = output_directory / "manifest.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Composition-only mode requires an existing manifest: {state_path}"
            )
        state = read_json(state_path)
        if state.get("status") != "complete":
            raise ValueError(f"Existing portfolio is not complete: {state_path}")
        if state.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError(
                "Existing portfolio checkpoint does not match --checkpoint"
            )
        if tuple(state.get("resolution", ())) != tuple(args.resolution_tuple):
            raise ValueError(
                "Composition resolution must match the completed captures: "
                f"{state.get('resolution')} != {list(args.resolution_tuple)}"
            )
        if int(state.get("raw_fps", 0)) != args.raw_fps:
            raise ValueError(
                "--raw-fps must match the completed captures: "
                f"{state.get('raw_fps')} != {args.raw_fps}"
            )
        state["recomposed_at"] = datetime.now().astimezone().isoformat()
        _compose_public_package(
            args,
            ffmpeg,
            ffprobe,
            output_directory,
            state,
            final_results,
            composition_mode="compose_only",
        )
        return

    adverse_selection_target = _adverse_selection_target(final_results_source)
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

        _compose_public_package(
            args,
            ffmpeg,
            ffprobe,
            output_directory,
            state,
            final_results,
            composition_mode="full_capture",
        )
        return

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
