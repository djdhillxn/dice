"""Overlay DICE command and hold diagnostics on a recorded MP4."""

import argparse
from pathlib import Path

import cv2
import pandas as pd


parser = argparse.ArgumentParser(description="Annotate a DICE video.")
parser.add_argument("--video", required=True)
parser.add_argument("--metrics", required=True)
parser.add_argument("--output", default="DICE_annotated.mp4")
args = parser.parse_args()

video_path = Path(args.video)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
metrics = pd.read_csv(args.metrics)
if metrics.empty:
    raise RuntimeError("The metrics CSV is empty.")

capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise RuntimeError("Could not open video: " + str(video_path))

fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_index = 0
while True:
    ok, frame = capture.read()
    if not ok:
        break

    row = metrics.iloc[min(frame_index, len(metrics) - 1)]
    target = int(row.target_face)
    top = int(row.top_face)
    alignment = float(row.alignment)
    hold = float(row.hold_progress)
    commands = int(row.commands_completed)

    cv2.rectangle(frame, (20, 20), (430, 185), (15, 15, 15), -1)
    cv2.putText(frame, "DICE", (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"TARGET FACE: {target}", (40, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    cv2.putText(frame, f"TOP FACE: {top}", (40, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (230, 230, 230), 2)
    cv2.putText(frame, f"ALIGNMENT: {alignment: .3f}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2)

    bar_left, bar_top, bar_width, bar_height = 455, 42, 260, 30
    cv2.rectangle(frame, (bar_left, bar_top), (bar_left + bar_width, bar_top + bar_height), (50, 50, 50), -1)
    cv2.rectangle(
        frame,
        (bar_left, bar_top),
        (bar_left + int(bar_width * max(0.0, min(1.0, hold))), bar_top + bar_height),
        (80, 220, 100),
        -1,
    )
    cv2.putText(frame, "HOLD", (bar_left, bar_top - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"COMMANDS: {commands}", (bar_left, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

    writer.write(frame)
    frame_index += 1

capture.release()
writer.release()
print("Saved", output_path.resolve())
