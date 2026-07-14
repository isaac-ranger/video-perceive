# DUAL CHANNEL: Take On Me (a-ha)

Source: [path]  
Interval: 0.5s  

## Motion grammar (guess)
- **label:** `music_video_montage` (confidence 0.85)
- **notes:** Captions are music marks/sparse lyrics — not VO disagreement. Picture edits may lock to a SOUND channel the instrument cannot hear yet. STROBE clusters = world thrash, not progressive cuts.
- **kinds:** {'OPEN': 1, 'HOLD': 56, 'JUMP_CUT': 143, 'LOCAL_MOVE': 226, 'STIR': 53, 'STROBE': 5, 'HARD_CHANGE': 4}
- **mean energy:** 27.07
- **centroid drift x (non-cut):** 0.0

## Speech ↔ picture coupling
- **mode:** `music_parallel`
- **disagree_score:** 0.315 · **cut_density_in_speech:** 0.315
- cuts_during_speech=94 / cuts_total=147 · speech_beats=298 · multi-cut speech runs=21

Kinds: `JUMP_CUT` = structural rewrite · `FADE` = exposure ramp, structure holds · `LOCAL_MOVE` = within-shot · `HOLD`/`STIR` = quiet · `HARD_CHANGE` = big rewrite (soft cut / gesture / ambiguous).

Channels stay separate. Words contextualize motion; do not blend unless you choose. High cut_density_in_speech + multi-cut caption runs ⇒ montage over VO.

---

## Windowed score

### 0:00–0:05
- **MOTION:** max_e=160.63 kind=JUMP_CUT mean_e=39.7 kinds={'OPEN': 1, 'HOLD': 1, 'JUMP_CUT': 3, 'LOCAL_MOVE': 4, 'STIR': 1}
- **SAID:** ♪♪

### 0:05–0:10
- **MOTION:** max_e=55.62 kind=LOCAL_MOVE mean_e=33.0 kinds={'LOCAL_MOVE': 4, 'HOLD': 1, 'JUMP_CUT': 4, 'STIR': 1}
- **SAID:** ♪♪

### 0:10–0:15
- **MOTION:** max_e=59.23 kind=JUMP_CUT mean_e=32.5 kinds={'JUMP_CUT': 6, 'HOLD': 2, 'STIR': 2}
- **SAID:** —

### 0:15–0:20
- **MOTION:** max_e=119.04 kind=JUMP_CUT mean_e=41.8 kinds={'LOCAL_MOVE': 3, 'JUMP_CUT': 4, 'HOLD': 3}
- **SAID:** —

### 0:20–0:25
- **MOTION:** max_e=58.43 kind=JUMP_CUT mean_e=37.1 kinds={'JUMP_CUT': 5, 'STIR': 2, 'STROBE': 1, 'LOCAL_MOVE': 2}
- **SAID:** —

### 0:25–0:30
- **MOTION:** max_e=49.7 kind=JUMP_CUT mean_e=28.0 kinds={'STIR': 3, 'JUMP_CUT': 3, 'LOCAL_MOVE': 3, 'HOLD': 1}
- **SAID:** —

### 0:30–0:35
- **MOTION:** max_e=64.86 kind=JUMP_CUT mean_e=50.5 kinds={'JUMP_CUT': 7, 'STIR': 1, 'STROBE': 2}
- **SAID:** —

### 0:35–0:40
- **MOTION:** max_e=71.12 kind=JUMP_CUT mean_e=41.4 kinds={'JUMP_CUT': 6, 'LOCAL_MOVE': 4}
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪

### 0:40–0:45
- **MOTION:** max_e=52.53 kind=JUMP_CUT mean_e=13.8 kinds={'LOCAL_MOVE': 3, 'JUMP_CUT': 2, 'HOLD': 5}
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪

### 0:45–0:50
- **MOTION:** max_e=63.47 kind=JUMP_CUT mean_e=20.8 kinds={'LOCAL_MOVE': 3, 'JUMP_CUT': 3, 'STIR': 4}
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪

### 0:50–0:55
- **MOTION:** max_e=85.6 kind=JUMP_CUT mean_e=48.7 kinds={'LOCAL_MOVE': 4, 'JUMP_CUT': 6}
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪
- **SAID:** ♪ [lyric redacted] ♪

### 0:55–1:00
- **MOTION:** max_e=67.86 kind=JUMP_CUT mean_e=30.4 kinds={'LOCAL_MOVE': 3, 'JUMP_CUT': 4, 'STIR': 3}
- **SAID:** ♪ [lyric redacted] ♪
