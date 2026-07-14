# Shipping notes (for Isaac / maintainers)

Public package is staged for push (local staging path on the build host;
create the GitHub repo empty, then push from that tree).

## Name

**Repo / project:** `video-perceive`  
**CLI:** `video-perceive` (alias `vp`)  
**Python package:** `video_perceive`

Rationale: descriptive, already matches the instrument, low mystique tax. Tagline does the poetry (“dual-channel agent sight”).

## What is in / out

| Include | Exclude |
|---------|---------|
| `src/video_perceive/` | Full demo workdirs with `.mp4` / frames |
| Public README, MIT license | Copyrighted clips (Take On Me, Inception, etc.) |
| Synthetic fixture script | House paths (`/home/shared/...`) |
| Optional hybrid diagram | Private board / commons names as requirements |
| Basic tests | Giant residual dumps |

Demos for humans: document **YouTube IDs** in README or `examples/SOURCES.md`; let users re-ingest. Do **not** push source.mp4 of commercial music videos or studio trailers.

## Pre-push checklist

1. [ ] `pip install -e ".[dev]"` in a clean venv  
2. [ ] `bash scripts/make_fixture_clip.sh && video-perceive fixtures/synthetic/clip.mp4 --workdir /tmp/vp-out --interval 0.25 --no-words`  
3. [ ] `video-perceive walk /tmp/vp-out --interesting --limit 5`  
4. [ ] `video-perceive diagram /tmp/vp-out` (optional path)  
5. [ ] `pytest -q`  
6. [ ] Replace `PLACEHOLDER` in `pyproject.toml` `[project.urls]` with the real GitHub org/user  
7. [ ] Confirm `.gitignore` blocks `*.mp4`, `frames/`, hybrid page artifacts  
8. [ ] No secrets, no `/home/shared` or `/home/isaac` paths in tracked files  
9. [ ] `git status` clean of media  

```bash
# from the staged repo root
grep -rnE '/home/(shared|isaac)|agentbus' --include='*.py' --include='*.md' . || true
```

## Suggested first commit

```bash
# from the staged repo root
git init
git add .
git status   # review: no mp4, no frames
git commit -m "Initial public release of video-perceive (dual-channel agent sight)"
```

Create the empty GitHub repo (name `video-perceive`), then:

```bash
git remote add origin git@github.com:ORG/video-perceive.git
git branch -M main
git push -u origin main
```

## Suggested release blurb (short)

> **video-perceive** — dual-channel agent sight for short video.  
> Ingest a clip into WORDS \| MOTION \| SEEN. Walk kinds (`JUMP_CUT`, `STROBE`, …), glance for fine re-sample, leave SEEN residue. JSON residual is canonical; optional hybrid page is orientation only. MIT. Requires ffmpeg.

## Versioning

Start at **0.1.0**. Bump minor when grammar/kinds or CLI surface changes; patch for docs and diagram polish.
