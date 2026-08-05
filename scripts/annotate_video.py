"""Overlay target and task diagnostics on a recorded DiceDial MP4."""

import argparse
from pathlib import Path

import cv2
import pandas as pd


parser = argparse.ArgumentParser(description="Annotate DiceDial video.")
parser.add_argument("--video", required=True)
parser.add_argument("--metrics", required=True)
parser.add_argument("--output", default="dicedial_annotated.mp4")
args = parser.parse_args()


video_path = Path(args.video)
metrics = pd.read_csv(args.metrics)
capture = cv2.VideoCapture(str(video_path))
if not capture.isOpened():
    raise RuntimeError("Could not open video: " + str(video_path))

fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    str(args.output),
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
    cv2.putText(frame, f"TARGET FACE: {target}", (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"TOP FACE: {top}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    cv2.putText(frame, f"ALIGNMENT: {alignment: .3f}", (40, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(frame, f"COMMANDS: {commands}", (40, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

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

    writer.write(frame)
    frame_index += 1

capture.release()
writer.release()
print("Saved", Path(args.output).resolve())
