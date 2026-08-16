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


def _align_rows(metrics, initial_metrics, frame_count):
    if frame_count == len(metrics):
        return metrics, "post_step"
    if frame_count == len(metrics) + 1:
        initial = {key: str(value) for key, value in initial_metrics.items()}
        return [initial, *metrics], "initial_plus_post_step"
    raise RuntimeError(
        "Video/telemetry synchronization failed: "
        f"video has {frame_count} frames, metrics has {len(metrics)} rows. "
        "Only exact or one-initial-frame alignment is supported."
    )


def _status_timeline(rows, raw_fps):
    statuses = [row.get("status", "rotating") for row in rows]
    completed_faces = [_integer(row, "completed_face") for row in rows]
    linger_frames = max(1, round(0.40 * raw_fps))
    for index, row in enumerate(rows):
        if _integer(row, "success") != 1:
            continue
        completed_face = _integer(row, "completed_face")
        for linger_index in range(index, min(len(rows), index + linger_frames)):
            if _integer(rows[linger_index], "drop"):
                break
            statuses[linger_index] = "confirmed"
            completed_faces[linger_index] = completed_face
    return statuses, completed_faces


def _panel(draw, box, radius, fill=(15, 20, 28, 218), outline=(255, 255, 255, 35)):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _text(draw, xy, text, font, fill=(245, 247, 250, 255), anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _annotate_frame(
    frame,
    row,
    status,
    completed_face,
    condition_label,
    style,
    fonts,
):
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
    top = _integer(row, "top_face")
    commands = _integer(row, "commands_completed")
    hold = max(0.0, min(1.0, _number(row, "hold_progress")))
    angle = _number(row, "angular_error_degrees")

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

    # Bottom diagnostic strip. The hero remains intentionally sparse.
    bottom_height = round((104 if style == "minimal" else 154) * scale)
    bottom_box = (
        margin,
        height - margin - bottom_height,
        width - margin,
        height - margin,
    )
    _panel(draw, bottom_box, radius)
    baseline = bottom_box[1] + round(27 * scale)
    _text(
        draw,
        (bottom_box[0] + round(22 * scale), baseline),
        f"COMMANDS  {commands}",
        fonts["body_bold"],
    )

    if style == "minimal":
        _text(
            draw,
            (bottom_box[2] - round(22 * scale), baseline),
            "20-DoF Shadow Hand  |  deterministic policy",
            fonts["body"],
            fill=(196, 204, 216, 255),
            anchor="ra",
        )
    else:
        _text(
            draw,
            (bottom_box[0] + round(340 * scale), baseline),
            f"TOP FACE  {top}",
            fonts["body_bold"],
        )
        _text(
            draw,
            (bottom_box[0] + round(610 * scale), baseline),
            f"ANGLE ERROR  {angle:4.1f} deg",
            fonts["body"],
            fill=(218, 224, 232, 255),
        )
        bar_left = bottom_box[0] + round(22 * scale)
        bar_right = bottom_box[2] - round(22 * scale)
        bar_top = bottom_box[1] + round(92 * scale)
        bar_bottom = bar_top + round(24 * scale)
        draw.rounded_rectangle(
            (bar_left, bar_top, bar_right, bar_bottom),
            radius=round(8 * scale),
            fill=(55, 62, 72, 255),
        )
        progress_right = bar_left + round((bar_right - bar_left) * hold)
        if progress_right > bar_left:
            draw.rounded_rectangle(
                (bar_left, bar_top, progress_right, bar_bottom),
                radius=round(8 * scale),
                fill=STATUS_COLORS["confirmed"]
                if hold >= 1.0
                else STATUS_COLORS["holding"],
            )
        _text(
            draw,
            (bar_right, bar_top - gap),
            f"HOLD  {round(20 * hold):02d} / 20",
            fonts["small_bold"],
            fill=(218, 224, 232, 255),
            anchor="ra",
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

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or summary.get("raw_fps", 60))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    aligned_rows, alignment_mode = _align_rows(metrics, initial_metrics, frame_count)
    statuses, completed_faces = _status_timeline(aligned_rows, raw_fps)

    scale = height / 1080.0
    fonts = {
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
        f"fps={output_fps}",
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
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = aligned_rows[decoded_frames]
            annotated = _annotate_frame(
                frame,
                row,
                statuses[decoded_frames],
                completed_faces[decoded_frames],
                condition_label,
                style,
                fonts,
            )
            process.stdin.write(annotated.tobytes())
            last_frame = annotated
            decoded_frames += 1

        if decoded_frames != frame_count:
            raise RuntimeError(
                f"Decoded {decoded_frames} frames but metadata advertised {frame_count}"
            )
        if last_frame is None:
            raise RuntimeError("The input video contained no decodable frames")
        for _ in range(round(post_roll_seconds * raw_fps)):
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
        "decoded_frames": decoded_frames,
        "post_roll_seconds": post_roll_seconds,
        "alignment_mode": alignment_mode,
        "crf": crf,
    }
    metadata_path = output_path.with_suffix(".annotation.json")
    metadata_path.write_text(
        json.dumps(annotation_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DICE] Saved annotated video: {output_path}")
    print(f"[DICE] Synchronization: {alignment_mode}, {decoded_frames} frames")
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
        "--style", choices=("minimal", "technical", "stress"), default="technical"
    )
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--post-roll-seconds", type=float, default=0.75)
    parser.add_argument("--font", default=None)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    values = parser.parse_args()
    if values.output_fps <= 0:
        parser.error("--output-fps must be positive")
    if not 0 <= values.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if values.post_roll_seconds < 0:
        parser.error("--post-roll-seconds must be non-negative")
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
        font_path=cli.font,
        ffmpeg_path=cli.ffmpeg,
    )
