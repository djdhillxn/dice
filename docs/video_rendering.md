# DICE portfolio video rendering

## Public deliverables

One completed training run produces exactly three presentation videos. Each
video has a same-named Markdown companion so the web page can render its own
HTML/CSS title, explanation, and quantitative caption.

| Export | Evidence shown | Views |
|---|---|---|
| `dice_nominal_success.mp4` | A representative six-command nominal rollout | Synchronized oblique and top-down views |
| `dice_physics_variation.mp4` | Nominal behavior beside a held-out ±20% mass/friction rollout | Full-height oblique views |
| `dice_adverse_boundary.mp4` | A representative heavy, slippery rollout through its eventual drop | Synchronized oblique and side-contact views |

All policy footage is presented at **0.5× real-time simulation speed**. Raw
captures remain 60 FPS; the exports are 30 FPS and retain the source frames,
so the slower presentation exposes finger motion instead of merely lowering
the frame rate. A 0.75-second terminal hold makes a confirmed face or drop
readable. Every MP4 is silent H.264, 1920x1080, `yuv420p`, and fast-start
optimized.

The public package also contains one footage-derived WebP poster and one
Markdown caption per video, SHA-256 checksums, and optional VP9 WebM
derivatives. It contains no rendered title/result cards and no contact sheet.
The page that embeds the assets owns the surrounding typography and layout.

## GCP preparation

Activate the training environment and install the presentation extra:

```bash
conda activate dice
cd ~/projects/dice
python -m pip install -e ".[video]"
```

The coordinator requires `ffmpeg`, `ffprobe`, an H.264 encoder, and DejaVu
fonts. Pillow supplies WebP poster encoding:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-dejavu-core

ffmpeg -hide_banner -encoders | grep -E 'libx264|libvpx-vp9'
ffprobe -version
```

OpenCV remains pinned to the Isaac Sim-compatible headless wheel in
`pyproject.toml`. Gymnasium and MoviePy create raw captures, Pillow draws the
dynamic HUD/legends, and FFmpeg performs the browser-compatible composition.

## First capture and composition

Pass the timestamped run ID below `outputs/`. `model_4000.pt`, headless mode,
1920x1080 raw capture, 60 raw FPS, 30 export FPS, and 0.5× playback are the
defaults:

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
  --export-fps 30 \
  --playback-speed 0.5
```

Add `--webm` for VP9 copies and `--show-ui` only on a machine with an
interactive display. A complete run/checkpoint directory is never silently
overwritten; full recapture requires `--force` and replaces that resolved
portfolio directory.

## Recompose existing captures without Isaac Sim

After pulling these code changes onto GCP, reuse the already completed scouts,
trajectories, telemetry, and camera captures:

```bash
python -u scripts/render_portfolio_videos.py \
  2026-08-16_11-22-43_angular_bound_pilot_gurgaon \
  --compose-only \
  --force
```

Composition-only mode does **not** start Isaac Sim, scout new seeds, or rerun
the policy. It verifies the existing manifest, checkpoint identity, capture
paths, resolution, FPS, and synchronized traces; stages all three new exports;
probes their technical contracts; and only then atomically replaces
`exports/`. If composition fails, the previous public package remains in
place. It can also rebase absolute GCP artifact paths after the complete
portfolio directory has been copied to another machine.

Legacy `cards/` or `intermediate/` directories from an earlier completed render
may remain as private provenance, but presentation revision 2 neither reads
nor publishes them. A fresh full render does not create static cards.

## Selection and evidence contract

The full coordinator:

1. Resolves the run and requires the matching completed final-evaluation
   checkpoint SHA-256.
2. Records repository and software provenance.
3. Audits the numbered presentation die against the stock evaluation cube,
   requiring matching mass and inertia within the declared tolerance.
4. Scouts fixed seed sets without cameras: nominal 7–11, symmetric variation
   17–21, and adverse 7–22.
5. Selects successful nominal and variation rollouts nearest their median
   six-command completion time—not the fastest or most flattering sample.
6. Derives the median failure commands/time from adverse evaluation episodes
   and selects the closest dropped scout.
7. Replays each selected 20-dimensional action trajectory through the required
   fixed camera views.
8. Requires exact command/success/drop event agreement and tight numeric trace
   agreement between scout and render replay.
9. Composes full-height center crops, one shared telemetry HUD for synchronized
   views, 0.5× playback, footage-derived posters, and Markdown captions.
10. Uses `ffprobe` to enforce codec, pixel format, dimensions, FPS, progressive
    loading, duration, and file-size contracts before installing exports.

The nominal oblique/top and adverse oblique/side panels are the same trajectory
and simulation ticks. The nominal/variation comparison intentionally shows two
separately selected median-like rollouts; it is a behavioral comparison, not a
claim of frame-identical physics counterfactuals. Aggregate 1,000-episode
evaluation values in the Markdown companions remain the generalization
evidence.

## Presentation conditions

| Condition | Task | Commands | Termination |
|---|---|---|---|
| Nominal | `DICE-Shadow-Play-v0` | Fixed cycle `1, 6, 3, 5, 2, 4` | Six confirmed commands, drop, or 40 s |
| Symmetric variation | `DICE-Shadow-Play-Robust-v0` | Same cycle | Six confirmed commands, drop, or 40 s |
| Adverse | `DICE-Shadow-Play-Adverse-v0` | Repeating cycle | Drop or the 24 s evaluation horizon |

All conditions use the same frozen deterministic policy and numbered die. The
symmetric condition samples object mass and physically consistent static and
dynamic friction within `[0.8, 1.2]` of nominal. The adverse condition fixes
mass at 1.5× and both friction coefficients at 0.7×. Presentation environments
defer only the final internal reset so the terminal confirmation or drop remains
visible; training and quantitative evaluation retain normal automatic resets.

## Output layout

```text
videos/<timestamped-run>_model_4000/
├── manifest.json
├── audit/
├── scout/<condition>/seed_<seed>/
├── captures/<condition>/seed_<seed>/<camera>/
└── exports/
    ├── dice_nominal_success.mp4
    ├── dice_nominal_success.md
    ├── dice_nominal_success_poster.webp
    ├── dice_physics_variation.mp4
    ├── dice_physics_variation.md
    ├── dice_physics_variation_poster.webp
    ├── dice_adverse_boundary.mp4
    ├── dice_adverse_boundary.md
    ├── dice_adverse_boundary_poster.webp
    └── checksums.sha256
```

`videos/` remains ignored by Git. Transfer the compact public package plus its
manifest from GCP:

```bash
scp -r \
  dheerajdhillon@dee-vm:~/projects/dice/videos/<run>_model_4000/exports \
  ./dice-portfolio-exports

scp \
  dheerajdhillon@dee-vm:~/projects/dice/videos/<run>_model_4000/manifest.json \
  ./dice-portfolio-exports/
```

Raw captures and working files do not belong in the GitHub Pages repository.

## GitHub Pages embedding

Use the nominal video as a silent inline loop:

```html
<video autoplay muted loop playsinline
       poster="assets/dice/dice_nominal_success_poster.webp"
       aria-describedby="dice-nominal-caption">
  <source src="assets/dice/dice_nominal_success.webm" type="video/webm">
  <source src="assets/dice/dice_nominal_success.mp4" type="video/mp4">
</video>
```

Use controls for the longer evidence clips:

```html
<video controls preload="metadata"
       poster="assets/dice/dice_adverse_boundary_poster.webp">
  <source src="assets/dice/dice_adverse_boundary.mp4" type="video/mp4">
</video>
```

Render the corresponding `.md` content as nearby page text; do not burn that
copy into new image cards. Git LFS is not used because GitHub Pages cannot
serve LFS objects.

## Low-level capture and annotation

The coordinator still builds on independently usable scripts. Capture one
view:

```bash
python -u scripts/play_rsl.py \
  --checkpoint outputs/<run>/model_4000.pt \
  --condition nominal \
  --camera hero \
  --seed 7 \
  --output videos/manual_nominal \
  --headless
```

Annotate it at half speed:

```bash
python scripts/annotate_video.py \
  --video videos/manual_nominal/raw/<capture>.mp4 \
  --metrics videos/manual_nominal/metrics.csv \
  --summary videos/manual_nominal/capture_summary.json \
  --initial-metrics videos/manual_nominal/initial_metrics.json \
  --style technical \
  --playback-speed 0.5 \
  --output videos/manual_nominal/annotated.mp4
```

Direct capture directories are immutable by design. Use a new destination
instead of mixing stale MP4 files with new telemetry.

## Failure handling

- **Physics audit mismatch:** do not bypass it for publication. The numbered
  visual asset must preserve the evaluated rigid-body behavior.
- **Replay trace mismatch:** the script stops because different trajectories
  must not be presented as synchronized camera views.
- **Video/telemetry mismatch:** only exact declared frame mappings and the
  known single missing terminal-frame MoviePy boundary case are accepted.
- **Composition-only validation failure:** keep the existing package intact;
  inspect the reported manifest, path, checkpoint, trace, or video contract.
- **Missing `libx264`:** install Ubuntu FFmpeg and verify the encoder.
- **Missing final evaluation:** run the final evaluation for this checkpoint;
  captions are never populated from another run.
- **Unexpected framing:** inspect the raw camera captures, adjust the named
  presets in `dicedial.portfolio_video`, and perform a full recapture. Center
  cropping in composition cannot repair a fundamentally misplaced camera.
