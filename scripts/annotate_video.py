"""Render a synchronized, resolution-independent HUD onto a DICE capture."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dicedial.portfolio_video import align_video_telemetry


STATUS_COLORS = {
    "rotating": (245, 176, 65, 255),
    "holding": (68, 196, 224, 255),
    "confirmed": (70, 211, 124, 255),
    "dropped": (242, 89, 89, 255),
}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _font_candidates(bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
        Path("/usr/share/fonts/truetype/liberation2")
        / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"),
    )


def _load_font(size, bold=False, explicit=None):
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(_font_candidates(bold=bold))
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _integer(row, key, default=0):
    value = row.get(key, default)
    return int(float(value))


def _number(row, key, default=0.0):
    value = row.get(key, default)
    return float(value)


def _count_decodable_frames(capture):
    """Count frames the decoder can actually consume, not container metadata."""

    frame_count = 0
    while capture.grab():
        frame_count += 1
    return frame_count


def _status_timeline(rows, raw_fps):
    statuses = [row.get("status", "rotating") for row in rows]
    completed_faces = [_integer(row, "completed_face") for row in rows]
    display_angles = [_number(row, "angular_error_degrees") for row in rows]
    linger_frames = max(1, round(0.40 * raw_fps))
    for index, row in enumerate(rows):
        if _integer(row, "success") != 1:
            continue
        completed_face = _integer(row, "completed_face")
        # The environment samples the next command on the success step, so its
        # logged angle already describes that new target. Retain the immediately
        # preceding completed-command angle during the editorial confirmation.
        completed_angle = display_angles[max(0, index - 1)]
        for linger_index in range(index, min(len(rows), index + linger_frames)):
            if _integer(rows[linger_index], "drop"):
                break
            statuses[linger_index] = "confirmed"
            completed_faces[linger_index] = completed_face
            display_angles[linger_index] = completed_angle
    return statuses, completed_faces, display_angles


def _panel(draw, box, radius, fill=(15, 20, 28, 218), outline=(255, 255, 255, 35)):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _text(draw, xy, text, font, fill=(245, 247, 250, 255), anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _annotate_frame(
    frame,
    row,
    status,
    completed_face,
    display_angle,
    condition_label,
    style,
    fonts,
    playback_speed=1.0,
    view_labels=(),
):
    if style == "none":
        return frame

    height, width = frame.shape[:2]
    scale = height / 1080.0
    margin = round(48 * scale)
    gap = round(14 * scale)
    radius = round(16 * scale)
    panel_height = round(124 * scale)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    target = _integer(row, "target_face")
    if status == "confirmed" and completed_face:
        target = completed_face
    top = _integer(row, "top_face")
    commands = _integer(row, "commands_completed")
    hold = max(0.0, min(1.0, _number(row, "hold_progress")))
    angle = float(display_angle)

    # Target badge.
    target_width = round((300 if style == "minimal" else 330) * scale)
    target_box = (margin, margin, margin + target_width, margin + panel_height)
    _panel(draw, target_box, radius)
    _text(
        draw,
        (target_box[0] + round(22 * scale), target_box[1] + round(20 * scale)),
        "TARGET FACE",
        fonts["small_bold"],
        fill=(176, 185, 198, 255),
    )
    _text(
        draw,
        (target_box[2] - round(46 * scale), target_box[1] + panel_height // 2),
        str(target),
        fonts["face"],
        anchor="mm",
    )

    # Condition/status badge.
    status_text = status.upper()
    if status == "confirmed" and completed_face:
        status_text = f"CONFIRMED FACE {completed_face}"
    condition_width = round(570 * scale)
    condition_box = (
        width - margin - condition_width,
        margin,
        width - margin,
        margin + panel_height,
    )
    _panel(draw, condition_box, radius)
    _text(
        draw,
        (condition_box[0] + round(22 * scale), condition_box[1] + round(20 * scale)),
        condition_label,
        fonts["small_bold"],
        fill=(176, 185, 198, 255),
    )
    _text(
        draw,
        (condition_box[0] + round(22 * scale), condition_box[1] + round(62 * scale)),
        status_text,
        fonts["body_bold"],
        fill=STATUS_COLORS.get(status, STATUS_COLORS["rotating"]),
    )

    # Playback speed is editorial metadata, not part of the physical condition.
    # Keep it in a dedicated compact chip so long condition labels (especially
    # the adverse case) never collide with or clip the condition/status panel.
    if playback_speed != 1.0:
        speed_width = round(220 * scale)
        speed_height = round(50 * scale)
        speed_right = condition_box[0] - gap
        speed_box = (
            speed_right - speed_width,
            margin,
            speed_right,
            margin + speed_height,
        )
        _panel(draw, speed_box, round(12 * scale), fill=(15, 20, 28, 185))
        _text(
            draw,
            (
                speed_box[0] + speed_width // 2,
                speed_box[1] + speed_height // 2,
            ),
            f"{playback_speed:g}× PLAYBACK",
            fonts["tiny_bold"],
            fill=(218, 224, 232, 255),
            anchor="mm",
        )

    # Camera labels describe synchronized panels without duplicating the HUD.
    if view_labels:
        label_y = margin + panel_height + round(18 * scale)
        panel_width = width / len(view_labels)
        chip_height = round(48 * scale)
        chip_width = round(210 * scale)
        for index, label in enumerate(view_labels):
            center_x = round((index + 0.5) * panel_width)
            chip_box = (
                center_x - chip_width // 2,
                label_y,
                center_x + chip_width // 2,
                label_y + chip_height,
            )
            _panel(draw, chip_box, round(12 * scale), fill=(15, 20, 28, 190))
            _text(
                draw,
                (center_x, label_y + chip_height // 2),
                label.upper(),
                fonts["small_bold"],
                fill=(226, 232, 240, 255),
                anchor="mm",
            )

    # Compact telemetry chips preserve the diagnostics without obscuring a
    # full-width strip of the hand and die.
    chip_height = round(68 * scale)
    chip_specs = (
        (round(285 * scale), f"COMMANDS  {commands}", fonts["body_bold"]),
        (round(245 * scale), f"TOP FACE  {top}", fonts["body_bold"]),
        (round(330 * scale), f"FACE ERROR  {angle:4.1f}°", fonts["body"]),
    )
    total_chip_width = sum(item[0] for item in chip_specs) + gap * (len(chip_specs) - 1)
    chip_left = (width - total_chip_width) // 2
    chip_top = height - margin - chip_height
    for chip_width, label, font in chip_specs:
        chip_box = (
            chip_left,
            chip_top,
            chip_left + chip_width,
            chip_top + chip_height,
        )
        _panel(draw, chip_box, round(13 * scale), fill=(15, 20, 28, 205))
        _text(
            draw,
            (chip_box[0] + round(20 * scale), chip_box[1] + chip_height // 2),
            label,
            font,
            fill=(226, 232, 240, 255),
            anchor="lm",
        )
        chip_left = chip_box[2] + gap

    # A narrow 20-segment rail is easier to read than a screen-wide progress
    # bar. On synchronized views it sits on the divider, leaving both subjects
    # unobstructed. The confirmed state is held by the existing status timeline.
    rail_width = round(84 * scale)
    rail_height = round(292 * scale)
    rail_center_x = width // 2 if view_labels else width - margin - rail_width // 2
    rail_top = margin + panel_height + round(104 * scale)
    rail_box = (
        rail_center_x - rail_width // 2,
        rail_top,
        rail_center_x + rail_width // 2,
        rail_top + rail_height,
    )
    _panel(draw, rail_box, round(13 * scale), fill=(15, 20, 28, 172))
    visual_hold = 1.0 if status == "confirmed" else hold
    hold_steps = max(0, min(20, round(20 * visual_hold)))
    _text(
        draw,
        (rail_center_x, rail_box[1] + round(24 * scale)),
        "HOLD",
        fonts["tiny_bold"],
        fill=(218, 224, 232, 255),
        anchor="mm",
    )
    _text(
        draw,
        (rail_center_x, rail_box[1] + round(52 * scale)),
        f"{hold_steps:02d}/20",
        fonts["tiny_bold"],
        fill=STATUS_COLORS.get(status, STATUS_COLORS["holding"]),
        anchor="mm",
    )
    segment_gap = max(1, round(2 * scale))
    segment_height = max(2, round(7 * scale))
    segment_width = round(32 * scale)
    segments_bottom = rail_box[3] - round(15 * scale)
    active_color = STATUS_COLORS.get(status, STATUS_COLORS["holding"])
    if status == "rotating":
        active_color = STATUS_COLORS["holding"]
    for segment in range(20):
        segment_bottom = segments_bottom - segment * (segment_height + segment_gap)
        segment_box = (
            rail_center_x - segment_width // 2,
            segment_bottom - segment_height,
            rail_center_x + segment_width // 2,
            segment_bottom,
        )
        draw.rounded_rectangle(
            segment_box,
            radius=max(1, round(2 * scale)),
            fill=active_color if segment < hold_steps else (62, 69, 79, 190),
        )

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def annotate_video(
    video_path,
    metrics_path,
    summary_path,
    initial_metrics_path,
    output_path,
    style="technical",
    output_fps=30,
    crf=16,
    post_roll_seconds=0.75,
    playback_speed=1.0,
    view_labels=(),
    font_path=None,
    ffmpeg_path="ffmpeg",
):
    """Annotate one capture and encode a browser-compatible H.264 MP4."""

    video_path = Path(video_path).expanduser().resolve()
    metrics_path = Path(metrics_path).expanduser().resolve()
    summary_path = Path(summary_path).expanduser().resolve()
    initial_metrics_path = Path(initial_metrics_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Annotated output already exists: {output_path}")

    ffmpeg = shutil.which(ffmpeg_path) or (
        str(Path(ffmpeg_path).resolve()) if Path(ffmpeg_path).is_file() else None
    )
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for H.264 portfolio encoding")

    metrics = _read_csv(metrics_path)
    if not metrics:
        raise RuntimeError("The metrics CSV is empty")
    summary = _read_json(summary_path)
    initial_metrics = _read_json(initial_metrics_path)

    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    raw_fps = float(probe.get(cv2.CAP_PROP_FPS) or summary.get("raw_fps", 60))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    advertised_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    decodable_frames = _count_decodable_frames(probe)
    probe.release()
    if decodable_frames == 0:
        raise RuntimeError("The input video contained no decodable frames")

    alignment = align_video_telemetry(
        metrics,
        initial_metrics,
        decodable_frames,
        frame_origin=summary.get("recording_frame_origin", "legacy_auto"),
    )
    frame_rows = alignment["frame_rows"]
    synthesized_rows = alignment["synthesized_rows"]
    alignment_mode = alignment["mode"]
    timeline_rows = [*frame_rows, *synthesized_rows]
    statuses, completed_faces, display_angles = _status_timeline(timeline_rows, raw_fps)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    scale = height / 1080.0
    fonts = {
        "tiny_bold": _load_font(max(14, round(20 * scale)), True, font_path),
        "small_bold": _load_font(max(16, round(25 * scale)), True, font_path),
        "body": _load_font(max(18, round(31 * scale)), False, font_path),
        "body_bold": _load_font(max(18, round(33 * scale)), True, font_path),
        "face": _load_font(max(34, round(68 * scale)), True, font_path),
    }
    condition_label = summary["condition_definition"]["label"]

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{raw_fps:.8f}",
        "-i",
        "-",
        "-vf",
        f"setpts=(PTS-STARTPTS)/{playback_speed:.8f},fps={output_fps}",
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
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    decoded_frames = 0
    synthesized_terminal_frames = 0
    last_source_frame = None
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_frames >= len(frame_rows):
                raise RuntimeError(
                    "The decoder produced more frames than the validated telemetry mapping"
                )
            row = frame_rows[decoded_frames]
            annotated = _annotate_frame(
                frame,
                row,
                statuses[decoded_frames],
                completed_faces[decoded_frames],
                display_angles[decoded_frames],
                condition_label,
                style,
                fonts,
                playback_speed,
                view_labels,
            )
            process.stdin.write(annotated.tobytes())
            last_source_frame = frame
            last_frame = annotated
            decoded_frames += 1

        if decoded_frames != decodable_frames:
            raise RuntimeError(
                f"Decoded {decoded_frames} frames after the validation pass found "
                f"{decodable_frames}"
            )
        if last_frame is None or last_source_frame is None:
            raise RuntimeError("The input video contained no decodable frames")

        for offset, row in enumerate(synthesized_rows):
            timeline_index = decoded_frames + offset
            annotated = _annotate_frame(
                last_source_frame.copy(),
                row,
                statuses[timeline_index],
                completed_faces[timeline_index],
                display_angles[timeline_index],
                condition_label,
                style,
                fonts,
                playback_speed,
                view_labels,
            )
            process.stdin.write(annotated.tobytes())
            last_frame = annotated
            synthesized_terminal_frames += 1

        # This many source-rate frames becomes the requested presentation hold
        # after the setpts playback-speed transform.
        post_roll_source_frames = round(post_roll_seconds * raw_fps * playback_speed)
        for _ in range(post_roll_source_frames):
            process.stdin.write(last_frame.tobytes())
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        process.wait()
        output_path.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()

    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg annotation failed ({return_code}): {stderr}")

    annotation_summary = {
        "schema_version": 1,
        "input_video": str(video_path),
        "metrics": str(metrics_path),
        "capture_summary": str(summary_path),
        "output_video": str(output_path),
        "style": style,
        "input_resolution": [width, height],
        "input_fps": raw_fps,
        "output_fps": output_fps,
        "playback_speed": playback_speed,
        "view_labels": list(view_labels),
        "advertised_frames": advertised_frames,
        "decodable_frames": decodable_frames,
        "decoded_frames": decoded_frames,
        "synthesized_terminal_frames": synthesized_terminal_frames,
        "telemetry_frames_encoded": decoded_frames + synthesized_terminal_frames,
        "post_roll_seconds": post_roll_seconds,
        "alignment_mode": alignment_mode,
        "crf": crf,
    }
    metadata_path = output_path.with_suffix(".annotation.json")
    metadata_path.write_text(
        json.dumps(annotation_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DICE] Saved annotated video: {output_path}")
    print(
        "[DICE] Synchronization: "
        f"{alignment_mode}, {decoded_frames} decoded + "
        f"{synthesized_terminal_frames} synthesized terminal frames "
        f"(container advertised {advertised_frames})"
    )
    return annotation_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate and H.264-encode one DICE portfolio capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--initial-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--style",
        choices=("none", "minimal", "technical", "stress"),
        default="technical",
    )
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--post-roll-seconds", type=float, default=0.75)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--view-label",
        action="append",
        default=[],
        help="Camera label; repeat once per synchronized view.",
    )
    parser.add_argument("--font", default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    values = parser.parse_args()
    if values.output_fps <= 0:
        parser.error("--output-fps must be positive")
    if not 0 <= values.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if values.post_roll_seconds < 0:
        parser.error("--post-roll-seconds must be non-negative")
    if values.playback_speed <= 0:
        parser.error("--playback-speed must be positive")
    return values


if __name__ == "__main__":
    cli = parse_args()
    annotate_video(
        cli.video,
        cli.metrics,
        cli.summary,
        cli.initial_metrics,
        cli.output,
        style=cli.style,
        output_fps=cli.output_fps,
        crf=cli.crf,
        post_roll_seconds=cli.post_roll_seconds,
        playback_speed=cli.playback_speed,
        view_labels=tuple(cli.view_label),
        font_path=cli.font,
        ffmpeg_path=cli.ffmpeg,
    )
