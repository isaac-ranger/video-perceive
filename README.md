# video-perceive

**Dual-channel agent sight for short video.**

Download (or point at) a clip. Agents read it as **separate channels** — never one fused soup:

| Channel | Contents |
|---------|----------|
| **WORDS** | SAID (captions) + optional SHOWN (OCR) |
| **MOTION** | energy, centroid, `frame_sim`, **kind** + **why/alt**, optional ASCII maps |
| **SEEN** | keyframe paths; **pairs at cuts**; optional `seen.jsonl` annotations |

Canonical residual is **JSON / jsonl / `score.md`**. An optional hybrid **page** (`page.json` + `page.png`) is only a projection for orientation.

## Why

Multimodal agents get lost when speech, cuts, and pixels are mashed into one narrative. This tool is an **ingest + walk** instrument:

1. **Ingest** a video into timed beats  
2. **Walk** the dual stream with filters  
3. **Glance** (fine re-sample) around interesting beats  
4. **See** (pay sight once; leave residue)

Motion is classified with an edit/action grammar (`JUMP_CUT` vs `LOCAL_MOVE` vs `STROBE` …) so jump-cut montages don’t read as “dancing.”

## Install

**System:** `ffmpeg` on `PATH`. Optional: `yt-dlp` (YouTube), `tesseract` (`--ocr` only).

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

# Directed glance — fine re-sample around a coarse beat
video-perceive glance ./out --around 42 --interval 0.1 --radius 1.0
# → glances/beat_0042/{frames,motion.jsonl,summary.json}

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

Guards: `sim ≥ 0.97` never promotes a cut (handheld/camera); missing frame stamp or duration mismatch forces re-extract.

Each beat may carry `kind_why`, `kind_alt` / `kind_alt_why`, `kind_confidence`, and `boundary_frames`. Cuts get `seen_pair` in `stream.jsonl`.

**Grammar** (`summary.json`):  
`music_video_montage` · `montage_over_speech` · `jump_cut_montage` · `continuous_rewrite` · `pedagogical_pulse` · `trajectory_or_scene` · `mixed`

**Coupling** modes:  
`music_parallel` · `montage_over_speech` · `speech_illustrates_motion` · `speech_with_action` · `motion_only`

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
