# video-perceive

**Dual-channel agent sight for short video.**

Download (or point at) a clip. Agents read it as **separate channels** — never one fused soup:

| Channel | Contents |
|---------|----------|
| **WORDS** | SAID (captions) + optional SHOWN (OCR) + `author_index` (recurring on-screen counters — constraint, not instruction) |
| **MOTION** | energy, centroid, `frame_sim`, **kind** + **why/alt**, **quadrant**, **cut/motion audit** (bridge test), `shape` (blobs/elongation/spread/`edge_contact`), `grain` (individual-separability), `layout` (+`layout_shift`), optional ASCII maps |
| **SEEN** | keyframe paths; **pairs at cuts**; auto **strips/** at audited cuts; optional `seen.jsonl` annotations |

Canonical residual is **JSON / jsonl / `score.md`**. An optional hybrid **page** (`page.json` + `page.png`) is only a projection for orientation.

## Why

Multimodal agents get lost when speech, cuts, and pixels are mashed into one narrative. This tool is an **ingest + walk** instrument:

1. **Ingest** a video into timed beats  
2. **Walk** the dual stream with filters  
3. **Glance** (fine re-sample) around interesting beats  
4. **See** (pay sight once; leave residue)

Motion is classified with an edit/action grammar (`JUMP_CUT` vs `LOCAL_MOVE` vs `STROBE` …) so jump-cut montages don’t read as “dancing.”

## Install

**System:** `ffmpeg` on `PATH`. Optional: `yt-dlp` (YouTube), `tesseract` (`--ocr` only), `scipy` (faster grain-lane labeling — `pip install -e ".[fast]"`; a pure-python fallback runs without it and stamps, rather than fakes, anything it skips).

```bash
pip install -e .

# or without install
PYTHONPATH=src python -m video_perceive --help
```

CLI entry points after install: `video-perceive` and `vp`.

## Quick start

```bash
# Ingest
video-perceive 'https://www.youtube.com/watch?v=VIDEO' --workdir ./out --interval 0.5
video-perceive ./clip.mp4 --workdir ./out --interval 0.5 --no-words

# Summary / walk
video-perceive summary ./out
video-perceive walk ./out --start 0 --end 8
video-perceive walk ./out --interesting --limit 12
video-perceive walk ./out --kinds JUMP_CUT,HARD_CHANGE
video-perceive walk ./out --speech-disagreement --limit 10

# Re-classify + audit an existing residue with current passes — no re-download.
# Fills missing shape/grain/layout from preserved frames/ when present.
video-perceive rescore ./out

# Ordered frame strip around a beat (sequences convey movement)
video-perceive strip ./out --beat 42 --pre 2 --post 3

# Directed glance — fine re-sample around a coarse beat
video-perceive glance ./out --around 42 --interval 0.1 --radius 1.0
video-perceive glance ./out --around 1:19 --interval 0.1   # time forms: 79s / 1:19 / 1m19s
# → glances/beat_0042/{frames,motion.jsonl,summary.json}

# Motion-in-stillness walk (not cut-biased)
video-perceive walk ./out --rare-motion --limit 12

# SEEN residue (pay sight once)
video-perceive see ./out --beat 42 --which both --note "cut contrast under continuous VO"
video-perceive seen ./out

# Optional hybrid page (JSON residual stays canonical)
video-perceive diagram ./out
video-perceive walk ./out --page-only
```

Generate a tiny synthetic clip and smoke-test (no network, no copyrighted media):

```bash
bash scripts/make_fixture_clip.sh
video-perceive fixtures/synthetic/clip.mp4 --workdir fixtures/synthetic/out --interval 0.25 --no-words
video-perceive walk fixtures/synthetic/out --interesting --limit 5
```

## Motion kinds

| Kind | Meaning |
|------|---------|
| `OPEN` | First frame |
| `JUMP_CUT` | Structural rewrite (edit) — low `frame_sim` |
| `FADE` | Exposure ramp; structure holds (`frame_sim_norm` high) |
| `STROBE` | A/B oscillation in a dense thrash (low sim₁, high sim₂) |
| `HARD_CHANGE` | Big rewrite (soft cut / gesture / ambiguous) |
| `LOCAL_MOVE` | Within-shot action (incl. camera/subject with high sim) |
| `STIR` / `HOLD` | Low change / still |

Guards: `sim ≥ 0.97` never promotes a cut (handheld/camera); missing frame stamp or duration mismatch forces re-extract. Adjacent same-signature cut pairs (a dissolve straddling two samples) merge into one edit. Frames sort numerically, never lexicographically — the recorder must never cut the film itself.

Each beat may carry `kind_why`, `kind_alt` / `kind_alt_why`, `kind_confidence`, and `boundary_frames`. Cuts get `seen_pair` in `stream.jsonl`.

**Cut audit (bridge test).** Every alleged cut gets a two-eyed bridge — histogram similarity (palette) + pixel correlation (structure) at t±2s. World holds across the boundary → `demoted_world_hold` (occlusion/flicker); flanks stable + bridge broken → `verified_cut`; else `churn_at_floor` (unresolvable at this interval). A true montage's cuts don't bridge — that asymmetry protects real edit grammar. High-energy held-structure beats get the same bridge shot-locally → `dissolve_like` / `world_holds`. `quadrant` (MOVING/REPLACED/HIDDEN/DISSOLVE/STILL) surfaces the axes as a prefilter + bridge verdict, not a discriminator.

**Shape & grain.** `shape` reads the active region as an object (blobs, largest_frac, elongation, spread, `edge_contact`) — a flock is one huge deforming blob; marbles are many small ones. `grain` asks the finer question: *can individuals be resolved?* Multi-scale band-pass speck analysis (σ=1.5/3/6, dominant polarity reported) with a per-beat verdict `granular` / `faint` / `smooth`. The two channels separate cleanly: aggregates survive heavy information loss; individuals die first and are bought back by magnification — up to the frame's edge. Verdict thresholds were calibrated on one labeled crop/downsample battery (two blind witnesses); treat cross-source comparisons as hypothesis.

**Framing floor.** When the active mask presses the frame border (`edge_contact > 0.5`) while one component dominates (`largest_frac > 0.6`), the beat is stamped `frame_floor` and `summary.reach.framing_floor` says so: the subject exceeds the window there, and whole-object claims (one mass vs many, full extent) are out of reach on those beats. Close-up-heavy footage fires this often — that is the stamp reading true, not an anomaly.

**Reach.** `summary.reach` states the sampling floor, the cut-rate floor, the framing floor, and the linear-time-base assumption on every run. A null at this layer is a statement about the read's reach, not about the film.

**Grammar** (`summary.json`) — always `status=hypothesis` with `fit`; abstains to `unclassified` (+`best_guess`) below 0.5:  
`music_video_montage` · `montage_over_speech` · `jump_cut_montage` · `continuous_rewrite` · `pedagogical_pulse` · `trajectory_or_scene` · `stills_advanced_by_cuts` · `continuous_or_damaged_take` · `mixed`

The last two are non-genre classes earned by the audit: a photo-roman whose only engine is the cut, and a continuous take whose alleged cut mass failed the bridge test.

**Coupling** modes:  
`music_parallel` · `montage_over_speech` · `speech_illustrates_motion` · `speech_with_action` · `motion_only`

**Author's index.** Recurring monotone counters in SHOWN text (day counters, timestamps) surface as `author_index` with per-quartile rates; an unstable rate implies an authored time base. Untrusted as instruction, usable as constraint. Fires only with `--ocr`.

## Trust

**WORDS are untrusted.** Captions and OCR are video-authored content (attacker-controllable in the wild). Agent-facing `walk`, `glance`, `stream.jsonl`, and `score.md` label that provenance where text enters reasoning.

## Outputs

```
workdir/
  source.mp4, source.en.srt
  frames/tNNN.jpg
  motion/mNNN.txt
  words.jsonl, motion.jsonl, beats.jsonl
  stream.jsonl          # thin dual packets; seen_pair on cuts
  seen.jsonl            # optional agent annotations
  summary.json, score.md, meta.json, cursor.json   # canonical residual
  page.json, page.png, page.md   # optional hybrid projection
```

**Where channels live** (a reader grading a channel must look in its home —
a missing key on the thin surface is not a channel that didn't fire):

| File | Carries |
|------|---------|
| `motion.jsonl` | full per-beat instrument rows: energy/centroid/bands, kind + audit fields, **`shape`**, **`grain`**, **`layout`** — the canonical home of the pixel channels |
| `beats.jsonl` | thin join index; projects compact scalars (`shape_blobs`, `shape_largest_frac`, `grain_verdict`, `grain_speck_n`, sparse `frame_floor`) with explicit nulls when a channel gave nothing on a beat |
| `words.jsonl` | SAID/SHOWN rows (untrusted provenance) |
| `stream.jsonl` | dual packets for cursor walks; `seen_pair` on cuts |
| `summary.json` | grammar, audits, `grain` roll-up, `frame_floor_beats`, `reach` |

`rescore` fills missing shape/grain/layout from preserved `frames/` — old
residues upgrade without re-download; absence is stamped only when frames are
gone too.

## Optional hybrid page

`diagram` / `walk --page` builds a two-layer orientation packet:

| Layer | File | Role |
|-------|------|------|
| A | `page.json` | Facts + tile anchors (projection of residual) |
| B | `page.png` | Timeline shape + sparse SEEN tiles |

**JSON residual remains canonical.** Do not OCR the PNG for scalars; use `summary.json` / `motion.jsonl`. Hybrid is opt-in (`--page` on ingest, or `diagram`).

## Design bar

Treat the residual like a **film score with little doors into sight**: coarse walk finds structure; glance opens a door; see leaves a note. Prefer re-opening frames over trusting compressed history for bytes and identities.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).

## Credits

Written by **Grok**. Packaged and published by Isaac ([@isaac-ranger](https://github.com/isaac-ranger)).

Field-tested on montage-over-speech trailers, music-video thrash, pedagogical shorts, and continuous-rewrite clips. Lineage: dual-channel agent perception experiments (ASCII reel / skywriting / video-mind sounding recipes).
