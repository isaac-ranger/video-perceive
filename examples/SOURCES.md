# Example sources (re-ingest yourself)

Do **not** commit commercial media. Re-run ingest against URLs you have rights to use.

| Label | Kind of tape | Example query |
|-------|----------------|---------------|
| Music video montage | Dense JUMP/STROBE thrash + lyrics | Search: a-ha “Take On Me” official |
| Montage over speech | One VO thought across many cuts | Search: Inception trailer |
| Pedagogical pulse | Hands / slides under speech | Educational shorts with on-screen demos |
| Pedagogical (lecture→IDE) | Long HOLD/STIR + one JUMP to screencast; speech labels holds | Search: “Passing JSON to an LLM is SUPER wasteful” (Matt Pocock short) |
| Continuous rewrite | Smooth camera / subject change | Single-take or soft-cut shorts |
| Trajectory | Clear path / travel | Shorts with simple subject motion |

**Field note (2026-07):** quiet screencasts often show modest centroid drift (PiP speaker, scroll). Grammar treats high quiet (`quiet_r ≥ 0.60`) as `pedagogical_pulse` even when `|drift| ≥ 0.12`, so they are not mislabeled as travel/scene.

```bash
video-perceive 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --workdir ./out --interval 0.5
video-perceive walk ./out --interesting --limit 12
```

Field note: music captions wrapped in `♪` are treated as music marks, not spoken VO disagreement.
