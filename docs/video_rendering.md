# DICE portfolio video rendering

## Deliverables

One completed training run produces exactly three public-facing videos:

| Export | Purpose | Content |
|---|---|---|
| `dice_hero.mp4` | Muted portfolio hero | Representative nominal six-face sequence with a minimal HUD |
| `dice_task_explainer.mp4` | Technical explanation | The same nominal trajectory from oblique, top, and side views, including a labeled half-speed hold replay |
| `dice_robustness_boundary.mp4` | Robustness and negative result | Nominal-versus-symmetric comparison followed by a representative adverse drop and labeled half-speed replay |

Every export is 1920x1080, 30 FPS, H.264, `yuv420p`, silent, and
fast-start optimized. The package also contains one WebP poster per video, a
three-camera contact sheet, SHA-256 checksums, and a complete rendering
manifest. Optional VP9 WebM derivatives can be requested with `--webm`.
Each MP4 is hard-limited to 50 MiB, and the manifest warns when their combined
payload exceeds the 40 MiB portfolio target.

## GCP preparation

Activate the same environment used for training and install the presentation
extra:

```bash
conda activate dice
cd ~/projects/dice
python -m pip install -e ".[video]"
```

The coordinator requires the system `ffmpeg` and `ffprobe` executables, an
H.264 encoder, and DejaVu fonts:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core

ffmpeg -hide_banner -encoders | grep -E 'libx264|libwebp|libvpx-vp9'
ffprobe -version
```

OpenCV remains pinned to the Isaac Sim-compatible wheel in `pyproject.toml`.
Gymnasium uses MoviePy for the raw capture; the pipeline uses Pillow for
anti-aliased HUD text and FFmpeg for the final browser-compatible encoding. It
does not use OpenCV's `mp4v` writer. The coordinator checks the Python modules
and required FFmpeg encoders before launching Isaac Sim.

## One-command render

Pass the timestamped run ID below `outputs/`. `model_4000.pt` is the default
checkpoint, and rendering is headless by default:

```bash
python -u scripts/render_portfolio_videos.py \
  2026-08-16_11-22-43_angular_bound_pilot_gurgaon
```

Equivalent explicit form:

```bash
python -u scripts/render_portfolio_videos.py \
  2026-08-16_11-22-43_angular_bound_pilot_gurgaon \
  --checkpoint model_4000.pt \
  --device cuda:0 \
  --resolution 1920x1080 \
  --raw-fps 60 \
  --export-fps 30
```

Add `--webm` to create VP9 alternatives. Add `--show-ui` only when rendering
on a machine with an interactive display. A completed output is never silently
overwritten; `--force` replaces only the resolved directory for this exact
run/checkpoint pair.

## What the coordinator does

1. Resolves the timestamped run and verifies the selected checkpoint plus its
   matching completed final-evaluation SHA-256.
2. Captures the repository commit, software versions, and checkpoint SHA-256.
3. Requires readable PhysX mass/inertia tensors, then compares the stock
   evaluation cube and numbered presentation die with a 2% relative tolerance
   and a `1e-8` absolute numerical floor.
4. Runs no-camera scouts for fixed seed sets:
   - nominal: seeds 7 through 11
   - symmetric variation: seeds 17 through 21
   - adverse: seeds 7 through 22
5. Selects the successful nominal and symmetric rollouts closest to the median
   six-command completion time.
6. Derives the median completed-command count and time-to-drop directly from
   the final adverse `episodes.csv`, then selects the closest dropped scout.
   For the completed `model_4000.pt` evaluation, those targets are nine
   commands and 7.28 seconds.
7. Saves the selected raw 20-dimensional action trajectories.
8. Replays the nominal trajectory from hero, top, and side cameras; the robust
   trajectory from the hero camera; and the adverse trajectory from hero and
   side cameras.
9. Requires exact command/success/drop event agreement and at most `1e-4`
   alignment, position, and hold-progress disagreement between scouting and
   rendered replays.
10. Applies the synchronized HUD, builds title/result cards, composes the three
    final stories, generates posters and the camera contact sheet, and writes
    checksums.
11. Uses `ffprobe` to require one silent H.264 stream, `yuv420p`, 1920x1080,
    30 FPS, metadata before media data for progressive loading, and less than
    50 MiB per MP4.

The process launches Isaac Sim sequentially for scouting and camera capture.
That is intentional: a single L4 renders one 1080p viewport at a time, avoiding
the memory and transfer cost of simultaneous camera render products.

## Presentation conditions

| Condition | Task | Commands | Termination |
|---|---|---|---|
| Nominal | `DICE-Shadow-Play-v0` | Fixed cycle `1, 6, 3, 5, 2, 4` | Six confirmed commands, drop, or 40 s |
| Symmetric variation | `DICE-Shadow-Play-Robust-v0` | Same cycle | Six confirmed commands, drop, or 40 s |
| Adverse | `DICE-Shadow-Play-Adverse-v0` | Repeating cycle | Drop or the 24 s evaluation horizon |

All three use one numbered die and one deterministic policy. The symmetric
condition samples object mass and material coefficients within `[0.8, 1.2]` of
nominal. The adverse condition fixes object mass at `1.5x` and static/dynamic
object friction at `0.7`.

Presentation environments defer only the final internal simulator reset. This
preserves the sixth confirmed hold or dropped state for the final captured
frame. Training and quantitative evaluation retain Isaac Lab's normal
automatic reset behavior.

## Output layout

```text
videos/<timestamped-run>_model_4000/
├── manifest.json
├── audit/
│   ├── stock/
│   ├── numbered/
│   ├── physics_snapshots.json
│   └── physics_audit.json
├── scout/<condition>/seed_<seed>/
│   ├── trajectory.npz
│   ├── metrics.csv
│   └── capture_summary.json
├── captures/<condition>/seed_<seed>/<camera>/
│   ├── raw/*.mp4
│   ├── metrics.csv
│   ├── initial_metrics.json
│   └── capture_summary.json
├── cards/
├── intermediate/
└── exports/
    ├── dice_hero.mp4
    ├── dice_hero_poster.webp
    ├── dice_task_explainer.mp4
    ├── dice_task_explainer_poster.webp
    ├── dice_robustness_boundary.mp4
    ├── dice_robustness_boundary_poster.webp
    ├── camera_contact_sheet.webp
    └── checksums.sha256
```

`videos/` remains ignored by Git. Transfer only the compact public package and
manifest from GCP:

```bash
scp -r \
  dheerajdhillon@dee-vm:~/projects/dice/videos/<run>_model_4000/exports \
  ./dice-portfolio-exports

scp \
  dheerajdhillon@dee-vm:~/projects/dice/videos/<run>_model_4000/manifest.json \
  ./dice-portfolio-exports/
```

Raw captures and intermediate files should not be committed or copied into the
GitHub Pages repository.

## GitHub Pages embedding

Use the hero as a silent inline loop:

```html
<video autoplay muted loop playsinline poster="assets/dice/dice_hero_poster.webp">
  <source src="assets/dice/dice_hero.webm" type="video/webm">
  <source src="assets/dice/dice_hero.mp4" type="video/mp4">
</video>
```

The two longer videos should not autoplay:

```html
<video controls preload="metadata"
       poster="assets/dice/dice_task_explainer_poster.webp">
  <source src="assets/dice/dice_task_explainer.mp4" type="video/mp4">
</video>
```

Keep only optimized exports and posters in the Pages repository. Git LFS is not
used because GitHub Pages cannot serve LFS objects.

## Low-level capture and annotation

The coordinator builds on two independently usable scripts. Capture one
nominal view:

```bash
python -u scripts/play_rsl.py \
  --checkpoint outputs/<run>/model_4000.pt \
  --condition nominal \
  --camera hero \
  --seed 7 \
  --output videos/manual_nominal \
  --headless
```

Annotate that capture:

```bash
python scripts/annotate_video.py \
  --video videos/manual_nominal/raw/<capture>.mp4 \
  --metrics videos/manual_nominal/metrics.csv \
  --summary videos/manual_nominal/capture_summary.json \
  --initial-metrics videos/manual_nominal/initial_metrics.json \
  --style technical \
  --output videos/manual_nominal/annotated.mp4
```

Direct capture directories are immutable by design. Use a new output directory
instead of mixing stale MP4 files and new telemetry.

## Failure handling

- **Physics audit mismatch:** do not bypass it for final publication. Compare
  `audit/physics_audit.json`; the numbered visual asset must not change the
  evaluated rigid-body behavior. `--skip-physics-audit` is for camera debugging
  only.
- **No adverse dropped scout:** rerun with a wider fixed range, for example
  `--adverse-seeds 7:40 --force`.
- **Replay trace mismatch:** camera rendering changed or exposed
  nondeterministic dynamics. The script intentionally stops rather than
  combining different trajectories as if they were synchronized views.
- **Video/telemetry mismatch:** the annotator counts actually decodable frames
  instead of trusting MP4 container metadata. It supports an exact post-step
  mapping, one explicitly declared initial frame, or one missing final frame
  when (and only when) the unmatched telemetry row is terminal. In that last
  MoviePy boundary case it repeats the final decoded image for one frame and
  overlays the true terminal telemetry. Capture summaries declare whether the
  recorder started from an initial or post-step frame, preventing an ambiguous
  one-frame offset from silently shifting every HUD label. Larger offsets and
  missing non-terminal frames remain hard failures; inspect the raw capture and
  Gymnasium version.
- **Missing `libx264`:** install the Ubuntu FFmpeg package and verify the
  encoder before rerunning.
- **Missing final evaluation:** the quantitative result cards are deliberately
  not populated from a different run or checkpoint. Run the final evaluation
  for the selected checkpoint, then rerun the renderer.
- **Unexpected framing:** review `exports/camera_contact_sheet.webp`, adjust the
  named camera presets in `dicedial.portfolio_video`, and rerender with
  `--force`.
