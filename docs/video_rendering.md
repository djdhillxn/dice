# DICE portfolio video rendering

## Public deliverables

One completed training run produces exactly three presentation videos. Each
video has a same-named Markdown companion so the web page can render its own
HTML/CSS title, explanation, and quantitative caption.

| Export | Evidence shown | Views |
|---|---|---|
| `dice_nominal_success.mp4` | One fixed-seed, 12-command nominal rollout | Synchronized oblique and top-down views |
| `dice_physics_variation.mp4` | Fixed-seed nominal behavior beside held-out ±20% mass/friction | Full-height oblique views |
| `dice_adverse_boundary.mp4` | The same declared seed under heavy, slippery physics through its drop | Synchronized oblique and side-contact views |

All policy footage is presented at **0.5× real-time simulation speed**. Raw
captures remain 60 FPS; the exports are 30 FPS and retain the source frames,
so the slower presentation exposes finger motion instead of merely lowering
the frame rate. A 0.75-second terminal hold makes a confirmed face or drop
readable. Every MP4 is silent H.264, 1920x1080, `yuv420p`, and fast-start
optimized.

The synchronized videos use floating telemetry chips rather than a wide bottom
banner. `FACE ERROR` is the angle between the requested face's outward normal
and world up, so 0° is perfect alignment. The narrow 20-segment `HOLD` rail
counts consecutive valid confirmation steps; alignment within 16° alone is not
enough because the position and angular-speed gates must also remain valid.

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
  2026-08-16_11-22-43_angular_bound_pilot_gurgaon \
  --force
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
  --playback-speed 0.5 \
  --seed 9 \
  --force
```

Add `--webm` for VP9 copies and `--show-ui` only on a machine with an
interactive display. A complete run/checkpoint directory is never silently
overwritten. `--force` replaces that exact resolved portfolio directory and
then reruns capture, replay, annotation, composition, and validation from the
beginning. It is the intended option while iterating on presentation code.

## One-command workflow and story registry

The public workflow is deliberately one end-to-end command. There is no
composition-only checkpoint. By default the script
renders exactly `nominal_success`, `physics_variation`, and `adverse_boundary`.
The small declarative registry in `dicedial.portfolio_video` supplies each
story's panels, cameras, HUD style, filename, and event-aware poster strategy.

For a later focused render, `--stories` accepts a comma-separated subset of
those keys and automatically derives the minimum camera plan. The standard
portfolio command should omit it so all three declared exports are produced.
This configurability does not introduce additional experiments.

## Capture and evidence contract

The full coordinator:

1. Resolves the run and requires the matching completed final-evaluation
   checkpoint SHA-256.
2. Records repository and software provenance.
3. Audits the numbered presentation die against the stock evaluation cube,
   requiring matching mass and inertia within the declared tolerance.
4. Uses representative seed `9` by default for every final presentation
   condition. The videos are illustrative; aggregate claims remain grounded in
   the completed 1,000-episode-per-condition evaluation.
5. Captures nominal and symmetric variation through exactly 12 confirmed
   commands. The adverse capture has no command limit and continues until its
   fixed-seed drop or the 24-second horizon; it fails clearly if no drop occurs.
6. Records each condition's primary 20-dimensional action trajectory and
   replays only that trace through the additional fixed camera view.
7. Requires exact command/success/drop event agreement and tight numeric trace
   agreement between the primary capture and every camera replay.
8. Composes full-height center crops, a compact shared telemetry HUD for
   synchronized views, 0.5× playback, footage-derived posters, and Markdown
   captions.
9. Uses `ffprobe` to enforce codec, pixel format, dimensions, FPS, progressive
   loading, duration, and file-size contracts before installing exports.

The nominal oblique/top and adverse oblique/side panels are the same trajectory
and simulation ticks. The nominal/variation comparison uses the same declared
seed but executes separately under different physics; it is a behavioral
comparison, not a claim of frame-identical physics counterfactuals. Aggregate
1,000-episode evaluation values in the Markdown companions remain the
generalization evidence.

## Presentation conditions

| Condition | Task | Commands | Termination |
|---|---|---|---|
| Nominal | `DICE-Shadow-Play-v0` | `1, 6, 3, 5, 2, 4, 6, 2, 5, 1, 3, 4` | 12 confirmed commands, drop, or 40 s |
| Symmetric variation | `DICE-Shadow-Play-Robust-v0` | Same 12-command sequence | 12 confirmed commands, drop, or 40 s |
| Adverse | `DICE-Shadow-Play-Adverse-v0` | Repeating original cycle `1, 6, 3, 5, 2, 4` | Drop or the 24 s evaluation horizon |

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
  --seed 9 \
  --command-limit 12 \
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
- **Fixed-seed nominal/variation failure:** inspect the capture rather than
  silently switching seeds; the declared 12-command evidence contract was not
  met.
- **Fixed-seed adverse horizon:** the script reports that the declared seed did
  not drop; it never searches for a more convenient failure.
- **Missing `libx264`:** install Ubuntu FFmpeg and verify the encoder.
- **Missing final evaluation:** run the final evaluation for this checkpoint;
  captions are never populated from another run.
- **Unexpected framing:** inspect the raw camera captures, adjust the named
  presets in `dicedial.portfolio_video`, and perform a full recapture. Center
  cropping in composition cannot repair a fundamentally misplaced camera.
