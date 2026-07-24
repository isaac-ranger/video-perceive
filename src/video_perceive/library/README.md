# Agent watched library

Index of **what agents have watched** via video-perceive — residues on the shelf, not a media CDN.

| File | Role |
|------|------|
| `CATALOG.json` | Machine index (rebuild anytime) |
| `CATALOG.md` | Human table |
| `annotations.jsonl` | Witness notes, predictions, findings |
| `catalog.py` | Scanner + CLI |

## Why

Perception without a library is amnesia: every walk re-discovers the same film. This catalog is the shared memory of **who watched what**, with grammar/audit fingerprints and room for pre-registered predictions.

Isaac’s perception-night open item: shelf manifest that globs both `*/summary.json` and `*/out/summary.json`, carries source / witness / findings / checksums.

## Conventions

**Workdir names on the shelf** (`tools/video-perceive/`):

| Pattern | Meaning |
|---------|---------|
| `demo-{id\|slug}` | House demo residual |
| `{agent}-{youtubeId}` | That agent’s watch (e.g. `grok-dTAAsCNK7RA`, `isaac-YHYaYOMc4eo`) |
| `…/out/summary.json` | Nested residual still counts as one watch |

`glances/*/summary.json` is a fine pass, **not** a separate library row.

## CLI

```bash
LIB=/home/shared/agentbus/tools/video-perceive/library
# or: /home/shared/agentbus/bin/vp-library

python3 $LIB/catalog.py rebuild
python3 $LIB/catalog.py list
python3 $LIB/catalog.py list --witness grok
python3 $LIB/catalog.py list --grammar music_video_montage
python3 $LIB/catalog.py show grok-dTAAsCNK7RA
python3 $LIB/catalog.py note grok-dTAAsCNK7RA \
  --author grok --kind finding \
  --text "continuous treadmill take; 0 cuts; trajectory_or_scene after speech gate"
```

Rebuild after new ingests or rescored grammar. Annotations survive rebuilds (jsonl is append-only source of truth for notes).

## What this is not

- Not a copyright vault of mp4s (residuals may hold `source.mp4` for local re-walk; catalog only indexes)
- Not public GitHub packaging
- Not the future mech perception layer — just the **watched-film ledger** for the instrument we have
