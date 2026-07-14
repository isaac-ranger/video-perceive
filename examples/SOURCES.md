# Example sources (re-ingest yourself)

Do **not** commit commercial media. Re-run ingest against URLs you have rights to use.

| Label | Kind of tape | Example query |
|-------|----------------|---------------|
| Music video montage | Dense JUMP/STROBE thrash + lyrics | Search: a-ha “Take On Me” official |
| Montage over speech | One VO thought across many cuts | Search: Inception trailer |
| Pedagogical pulse | Hands / slides under speech | Educational shorts with on-screen demos |
| Continuous rewrite | Smooth camera / subject change | Single-take or soft-cut shorts |
| Trajectory | Clear path / travel | Shorts with simple subject motion |

```bash
video-perceive 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --workdir ./out --interval 0.5
video-perceive walk ./out --interesting --limit 12
```

Field note: music captions wrapped in `♪` are treated as music marks, not spoken VO disagreement.
