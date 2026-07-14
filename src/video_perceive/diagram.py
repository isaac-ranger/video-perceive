#!/usr/bin/env python3
"""Optional hybrid residual page: Layer A (JSON) + Layer B (image SEEN/shape).

Canonical truth remains the workdir residual:
  summary.json, score.md, motion.jsonl, stream.jsonl, words.jsonl, frames/

This page is a *projection* for orientation — not a second source of truth.

  A  page.json  — numbers/kinds/handles (copied from the residual; prefer residual)
  B  page.png   — timeline shape + appearance; do NOT OCR for scalars
  ·  page.md    — agent read-order (thin)

Image is rendered *from* Layer A so the two cannot drift.

  python -m video_perceive diagram ./workdir
  video-perceive diagram ./workdir
  video-perceive walk ./workdir --page-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# --- palette ------------------------------------------------------------------
BG = (12, 14, 18)
PANEL = (22, 26, 34)
INK = (220, 224, 230)
MUTED = (140, 148, 160)
DIM = (80, 88, 100)
ACCENT = (120, 200, 255)
WARN = (240, 200, 80)
LAYER_A = (100, 210, 140)
LAYER_B = (120, 180, 255)

KIND_COLOR = {
    "OPEN": (240, 240, 240),
    "HOLD": (55, 60, 70),
    "STIR": (90, 100, 120),
    "LOCAL_MOVE": (60, 180, 200),
    "FADE": (90, 130, 220),
    "JUMP_CUT": (220, 70, 70),
    "HARD_CHANGE": (230, 140, 50),
    "STROBE": (220, 60, 200),
}

FONT_REG = "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf"

SCHEMA = "video-perceive.hybrid_page/v1"
CUT_KINDS = frozenset({"JUMP_CUT", "HARD_CHANGE", "STROBE", "FADE"})


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def pick_salient(motion: list[dict], n: int = 8) -> list[dict]:
    priority = {"STROBE": 4, "JUMP_CUT": 3, "HARD_CHANGE": 3, "FADE": 2, "LOCAL_MOVE": 1}
    scored: list[tuple[float, dict]] = []
    for m in motion:
        kind = m.get("kind") or ""
        if kind in ("HOLD", "STIR", "OPEN"):
            continue
        e = float(m.get("energy") or 0)
        conf = float(m.get("kind_confidence") or 0.5)
        score = e * (1.0 + 0.5 * priority.get(kind, 0)) * (0.7 + 0.3 * conf)
        if m.get("kind_alt"):
            score *= 1.15
        if conf < 0.75:
            score *= 1.1  # ambiguity wants SEEN
        scored.append((score, m))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return []
    duration = max(float(m.get("t") or 0) for m in motion) or 1.0
    min_gap = duration / max(n * 1.5, 1)
    picked: list[dict] = []
    for _, m in scored:
        t = float(m.get("t") or 0)
        if any(abs(t - float(p.get("t") or 0)) < min_gap for p in picked):
            continue
        picked.append(m)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for _, m in scored:
            if m not in picked:
                picked.append(m)
            if len(picked) >= n:
                break
    picked.sort(key=lambda m: float(m.get("t") or 0))
    return picked


def is_ambiguous(m: dict) -> bool:
    conf = float(m.get("kind_confidence") or 1.0)
    return bool(m.get("kind_alt")) or conf < 0.75


def load_tile(workdir: Path, frame_name: str | None, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    blank = Image.new("RGB", size, (30, 34, 42))
    if not frame_name:
        return blank
    path = workdir / "frames" / frame_name
    if not path.exists():
        return blank
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (20, 22, 28))
        canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
        return canvas
    except OSError:
        return blank


# --- Layer A: structured hybrid document --------------------------------------

def build_layer_a(
    workdir: Path,
    tiles_n: int = 8,
    handles_n: int = 10,
    timeline_samples: int = 160,
) -> dict[str, Any]:
    """Authoritative facts + anchors. Image is painted from this doc."""
    workdir = Path(workdir)
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    motion = load_jsonl(workdir / "motion.jsonl")
    words = load_jsonl(workdir / "words.jsonl")
    words_by_beat = {w.get("beat"): w for w in words}
    if not motion:
        raise SystemExit(f"no motion.jsonl in {workdir}")

    interval = summary.get("interval")
    if interval is None and (workdir / "meta.json").exists():
        interval = json.loads((workdir / "meta.json").read_text()).get("interval")

    grammar = summary.get("grammar") or {}
    coupling = summary.get("coupling") or {}
    kind_counts = summary.get("kind_counts") or {}
    n_beats = int(summary.get("n_beats") or len(motion))
    t_max = float(motion[-1].get("t") or 0) or 1.0

    # downsample full motion into a coarse timeline (shape without 344k jsonl)
    step = max(1, len(motion) // timeline_samples)
    timeline = []
    for m in motion[::step]:
        w = words_by_beat.get(m.get("beat")) or {}
        said = (w.get("said") or "").strip()
        speech = None
        if said:
            speech = "music" if set(said) <= set("♪♫ ") or said in ("♪♪", "♪") else "speech"
        timeline.append({
            "beat": m.get("beat"),
            "t": m.get("t"),
            "kind": m.get("kind"),
            "energy": m.get("energy"),
            "conf": m.get("kind_confidence"),
            "speech": speech,
        })

    salient = pick_salient(motion, n=tiles_n)
    tiles = []
    for i, m in enumerate(salient):
        kind = m.get("kind") or "?"
        from_f = m.get("from_frame")
        to_f = m.get("to_frame") or (m.get("boundary_frames") or [None])[-1]
        w = words_by_beat.get(m.get("beat")) or {}
        amb = is_ambiguous(m)
        tile = {
            "id": f"T{i}",
            "beat": m.get("beat"),
            "t": m.get("t"),
            "clock": m.get("clock"),
            "kind": kind,
            "kind_why": m.get("kind_why"),
            "kind_alt": m.get("kind_alt"),
            "kind_alt_why": m.get("kind_alt_why"),
            "kind_confidence": m.get("kind_confidence"),
            "energy": m.get("energy"),
            "frame_sim": m.get("frame_sim"),
            "from_frame": from_f,
            "to_frame": to_f,
            "pair": bool(kind in CUT_KINDS and from_f and to_f),
            "said": w.get("said"),
            "said_trust": w.get("trust") or summary.get("words_trust") or "untrusted",
            "ambiguous": amb,
            "resolve": (
                f"glance --around {m.get('beat')}"
                if amb
                else f"see --beat {m.get('beat')} or open frames"
            ),
            # image geometry filled after render
            "bbox": None,
        }
        tiles.append(tile)

    handles = []
    for m in pick_salient(motion, n=handles_n):
        w = words_by_beat.get(m.get("beat")) or {}
        handles.append({
            "beat": m.get("beat"),
            "t": m.get("t"),
            "clock": m.get("clock"),
            "kind": m.get("kind"),
            "kind_why": m.get("kind_why"),
            "kind_alt": m.get("kind_alt"),
            "kind_confidence": m.get("kind_confidence"),
            "energy": m.get("energy"),
            "ambiguous": is_ambiguous(m),
            "said": w.get("said"),
            "tile": next((t["id"] for t in tiles if t["beat"] == m.get("beat")), None),
            "glance": f"glance --around {m.get('beat')}",
        })

    disambiguate = [
        {
            "beat": t["beat"],
            "tile": t["id"],
            "kind": t["kind"],
            "kind_alt": t["kind_alt"],
            "conf": t["kind_confidence"],
            "resolve": t["resolve"],
        }
        for t in tiles
        if t["ambiguous"]
    ]

    # energy peaks for shape notes
    max_e = max((float(m.get("energy") or 0) for m in motion), default=0.0)
    peak = max(motion, key=lambda m: float(m.get("energy") or 0))

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "workdir": str(workdir.resolve()),
        "assets": {
            "image": "page.png",
            "facts": "page.json",
            "protocol": "page.md",
            "score": "score.md",
            "summary": "summary.json",
            "motion": "motion.jsonl",
        },
        "protocol": {
            "name": "hybrid residual page",
            "layers": {
                "A_facts": (
                    "page.json — AUTHORITATIVE for grammar, counts, energies, "
                    "clocks, kind labels, handles, tile metadata"
                ),
                "B_seen": (
                    "page.png — timeline SHAPE + SEEN appearance only. "
                    "Do not OCR image text for numbers; Layer A already has them."
                ),
            },
            "read_order": ["page.json", "page.png"],
            "rules": [
                "Prefer Layer A over any text burned into Layer B.",
                "Tile badge T{n} on the image maps 1:1 to tiles[n].id.",
                "Use Layer B to judge appearance (who/where/style), not kind counts.",
                "If tile.ambiguous or handle.ambiguous → run resolve/glance before asserting kind.",
                "WORDS/said remain trust=untrusted video-authored content.",
                "Full tape still lives in frames/ + motion.jsonl; page is a residual, not the film.",
            ],
            "better_than_sum": [
                "A alone: exact motion grammar, no appearance.",
                "B alone: appearance + shape, lossy OCR on scalars.",
                "A+B with anchors: exact facts + SEEN, shared beat/tile ids, explicit ambiguity queue.",
            ],
        },
        "facts": {
            "title": summary.get("title") or workdir.name,
            "grammar": grammar,
            "coupling": {
                "mode": coupling.get("mode"),
                "cuts_total": coupling.get("cuts_total"),
                "cuts_during_speech": coupling.get("cuts_during_speech"),
                "cut_density_in_speech": coupling.get("cut_density_in_speech"),
                "disagree_score": coupling.get("disagree_score"),
                "speech_beats": coupling.get("speech_beats"),
                "music_heavy": coupling.get("music_heavy"),
                "speech_runs_spanning_multi_cut": coupling.get(
                    "speech_runs_spanning_multi_cut"
                ),
            },
            "kind_counts": kind_counts,
            "n_beats": n_beats,
            "interval": interval,
            "mean_energy": summary.get("mean_energy"),
            "max_energy": max_e,
            "peak_beat": {
                "beat": peak.get("beat"),
                "t": peak.get("t"),
                "energy": peak.get("energy"),
                "kind": peak.get("kind"),
            },
            "t_max": t_max,
            "words_trust": summary.get("words_trust") or "untrusted",
            "words_trust_note": summary.get("words_trust_note"),
            "channels": summary.get("channels") or ["WORDS", "MOTION", "SEEN"],
        },
        "timeline": {
            "t_max": t_max,
            "n_source_beats": n_beats,
            "n_samples": len(timeline),
            "samples": timeline,
            "note": "Coarse shape for agents; use motion.jsonl for full resolution.",
        },
        "tiles": tiles,
        "handles": handles,
        "disambiguate": disambiguate,
        "kind_legend": {k: list(v) for k, v in KIND_COLOR.items()},
    }
    return doc


# --- Layer B: render image from Layer A ---------------------------------------

def render_layer_b(
    workdir: Path,
    doc: dict[str, Any],
    width: int = 1600,
    height: int = 1200,
) -> Image.Image:
    workdir = Path(workdir)
    facts = doc["facts"]
    tiles = doc["tiles"]
    handles = doc["handles"]
    samples = doc["timeline"]["samples"]
    kind_counts = facts.get("kind_counts") or {}
    coupling = facts.get("coupling") or {}
    grammar = facts.get("grammar") or {}
    t_max = float(facts.get("t_max") or 1.0) or 1.0

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    f10, f11, f12 = font(10), font(11), font(12)
    f14b, f16b, f20b = font(14, True), font(16, True), font(20, True)

    pad = 16
    y = pad

    def text(xy, s, fnt=f11, fill=INK):
        draw.text(xy, s, font=fnt, fill=fill)

    # protocol banner
    draw.rounded_rectangle([pad, y, width - pad, y + 36], radius=6, fill=PANEL)
    text((pad + 10, y + 4), "HYBRID PAGE  ·  Layer A=JSON facts (authority)  ·  Layer B=image SEEN/shape", f12, ACCENT)
    text(
        (pad + 10, y + 20),
        "Do not OCR numbers from this image — read page.json. Tile badges T0.. map to tiles[].",
        f10,
        MUTED,
    )
    y += 48

    # header from Layer A facts
    draw.rounded_rectangle([pad, y, width - pad, y + 64], radius=8, fill=PANEL)
    text((pad + 12, y + 6), truncate(str(facts.get("title")), 70), f20b, LAYER_A)
    g_label = grammar.get("label", "?")
    g_conf = grammar.get("confidence")
    mode = coupling.get("mode", "?")
    line2 = (
        f"A│ grammar={g_label} conf={g_conf}  mode={mode}  beats={facts.get('n_beats')}  "
        f"interval={facts.get('interval')}s  mean_e={facts.get('mean_energy')}"
    )
    text((pad + 12, y + 34), truncate(line2, 120), f11, MUTED)
    notes = grammar.get("notes") or ""
    if notes:
        text((pad + 12, y + 48), truncate(f"note: {notes}", 120), f10, DIM)
    y += 76

    # histogram + coupling
    left_w = int(width * 0.55) - pad
    right_x = pad + left_w + 12
    right_w = width - pad - right_x
    hist_h = 100
    draw.rounded_rectangle([pad, y, pad + left_w, y + hist_h], radius=8, fill=PANEL)
    draw.rounded_rectangle([right_x, y, right_x + right_w, y + hist_h], radius=8, fill=PANEL)
    text((pad + 10, y + 6), "A│ KIND COUNTS", f11, LAYER_A)
    text((right_x + 10, y + 6), "A│ COUPLING / TRUST", f11, LAYER_A)

    order = ["JUMP_CUT", "HARD_CHANGE", "STROBE", "FADE", "LOCAL_MOVE", "STIR", "HOLD", "OPEN"]
    present = [k for k in order if kind_counts.get(k)]
    for k in kind_counts:
        if k not in present:
            present.append(k)
    max_c = max((int(kind_counts.get(k, 0)) for k in present), default=1) or 1
    bar_top = y + 26
    bar_h = hist_h - 38
    bar_w = max(14, (left_w - 24) // max(len(present), 1) - 6)
    for i, k in enumerate(present):
        c = int(kind_counts.get(k, 0))
        bh = int(bar_h * (c / max_c))
        bx = pad + 12 + i * (bar_w + 6)
        by = bar_top + bar_h - bh
        draw.rectangle([bx, by, bx + bar_w, bar_top + bar_h], fill=KIND_COLOR.get(k, (150, 150, 150)))
        text((bx, bar_top + bar_h + 2), f"{c}", f10, INK)

    coup_lines = [
        f"cuts_total={coupling.get('cuts_total')}",
        f"cuts_during_speech={coupling.get('cuts_during_speech')}",
        f"cut_density_in_speech={coupling.get('cut_density_in_speech')}",
        f"disagree_score={coupling.get('disagree_score')}",
        f"speech_beats={coupling.get('speech_beats')} music_heavy={coupling.get('music_heavy')}",
        f"multi_cut_runs={coupling.get('speech_runs_spanning_multi_cut')}",
        f"WORDS trust={facts.get('words_trust')}",
    ]
    for i, line in enumerate(coup_lines):
        text((right_x + 10, y + 24 + i * 11), truncate(line, 48), f10, INK)
    y += hist_h + 10

    # timelines from samples (Layer A timeline → paint)
    tl_h = 140
    draw.rounded_rectangle([pad, y, width - pad, y + tl_h], radius=8, fill=PANEL)
    text((pad + 10, y + 4), "B│ TIME shape (kinds · energy · speech) — proportions visual; counts live in A", f11, LAYER_B)

    x0, x1 = pad + 10, width - pad - 10
    usable_w = x1 - x0

    def x_at(t: float) -> int:
        return x0 + int(usable_w * clamp01(float(t) / t_max))

    strip_y = y + 22
    strip_h = 26
    draw.rectangle([x0, strip_y, x1, strip_y + strip_h], fill=(18, 20, 26))
    if samples:
        # paint with sample width
        for i, s in enumerate(samples):
            t = float(s.get("t") or 0)
            t_next = float(samples[i + 1]["t"]) if i + 1 < len(samples) else t_max
            wpx = max(1, x_at(t_next) - x_at(t))
            col = KIND_COLOR.get(s.get("kind") or "", (100, 100, 100))
            xx = x_at(t)
            draw.rectangle([xx, strip_y, min(x1, xx + wpx), strip_y + strip_h], fill=col)

    e_y = strip_y + strip_h + 8
    e_h = 36
    draw.rectangle([x0, e_y, x1, e_y + e_h], fill=(18, 20, 26))
    max_e = float(facts.get("max_energy") or 1.0) or 1.0
    pts = []
    for s in samples:
        e = float(s.get("energy") or 0)
        pts.append((x_at(float(s.get("t") or 0)), e_y + e_h - int((e / max_e) * (e_h - 2)) - 1))
    if len(pts) >= 2:
        draw.line(pts, fill=ACCENT, width=1)

    s_y = e_y + e_h + 6
    s_h = 14
    draw.rectangle([x0, s_y, x1, s_y + s_h], fill=(18, 20, 26))
    for s in samples:
        sp = s.get("speech")
        if not sp:
            continue
        xx = x_at(float(s.get("t") or 0))
        col = (100, 80, 160) if sp == "music" else (180, 180, 90)
        draw.rectangle([xx, s_y + 2, min(x1, xx + 3), s_y + s_h - 2], fill=col)

    # mark tile times on strip
    for t in tiles:
        xx = x_at(float(t.get("t") or 0))
        draw.line([(xx, strip_y - 2), (xx, s_y + s_h)], fill=WARN if t.get("ambiguous") else LAYER_B, width=1)
        text((xx - 6, strip_y - 12), t["id"], f10, WARN if t.get("ambiguous") else LAYER_B)

    tick_y = s_y + s_h + 4
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tt = t_max * frac
        xx = x_at(tt)
        text((xx - 10, tick_y), f"{int(tt // 60)}:{int(tt % 60):02d}", f10, DIM)

    leg_y = y + tl_h - 14
    lx = pad + 10
    for k in order:
        if k not in kind_counts and k not in ("JUMP_CUT", "LOCAL_MOVE", "HOLD"):
            continue
        draw.rectangle([lx, leg_y, lx + 9, leg_y + 9], fill=KIND_COLOR.get(k, (100, 100, 100)))
        text((lx + 12, leg_y - 1), k, f10, MUTED)
        lx += 14 + 7 * len(k) + 10
    y += tl_h + 10

    # SEEN tiles
    tile_panel_h = 300
    draw.rounded_rectangle([pad, y, width - pad, y + tile_panel_h], radius=8, fill=PANEL)
    text(
        (pad + 10, y + 6),
        f"B│ SEEN · {len(tiles)} tiles · yellow badge = ambiguous (use A.resolve / glance) · pair = before/after",
        f11,
        LAYER_B,
    )

    if tiles:
        cell_w = (width - 2 * pad - 20) // len(tiles)
        cell_h = tile_panel_h - 34
        thumb_h = cell_h - 52
        for i, tmeta in enumerate(tiles):
            cx = pad + 10 + i * cell_w
            cy = y + 24
            kind = tmeta.get("kind") or "?"
            col = KIND_COLOR.get(kind, (150, 150, 150))
            border = WARN if tmeta.get("ambiguous") else col
            draw.rectangle([cx, cy, cx + cell_w - 6, cy + cell_h - 8], outline=border, width=2)

            # bbox in image coords for Layer A
            tmeta["bbox"] = {
                "x0": cx,
                "y0": cy,
                "x1": cx + cell_w - 6,
                "y1": cy + cell_h - 8,
                "page_size": [width, height],
            }

            from_f, to_f = tmeta.get("from_frame"), tmeta.get("to_frame")
            if tmeta.get("pair") and from_f and to_f:
                th = (thumb_h - 4) // 2
                tw = cell_w - 14
                img.paste(load_tile(workdir, from_f, (tw, th)), (cx + 4, cy + 4))
                img.paste(load_tile(workdir, to_f, (tw, th)), (cx + 4, cy + 8 + th))
            else:
                img.paste(load_tile(workdir, to_f, (cell_w - 14, thumb_h)), (cx + 4, cy + 4))

            # large tile id badge (cross-anchor)
            badge = tmeta["id"]
            bw, bh = 28, 16
            draw.rectangle([cx + 4, cy + 4, cx + 4 + bw, cy + 4 + bh], fill=(0, 0, 0))
            text((cx + 8, cy + 5), badge, f11, WARN if tmeta.get("ambiguous") else LAYER_B)

            ly = cy + cell_h - 46
            # labels are hints; authority is A
            text((cx + 4, ly), truncate(f"{badge} b{tmeta.get('beat')} @{tmeta.get('clock')} {kind}", max(14, cell_w // 7)), f10, INK)
            line2 = f"e={tmeta.get('energy')} conf={tmeta.get('kind_confidence')}"
            if tmeta.get("kind_alt"):
                line2 += f" alt={tmeta.get('kind_alt')}"
            text((cx + 4, ly + 12), truncate(line2, max(14, cell_w // 7)), f10, MUTED)
            text((cx + 4, ly + 24), truncate(tmeta.get("kind_why") or "", max(14, cell_w // 7)), f10, DIM)

    y += tile_panel_h + 10

    # handles from A
    bot_h = max(90, height - y - pad)
    draw.rounded_rectangle([pad, y, width - pad, y + bot_h], radius=8, fill=PANEL)
    text((pad + 10, y + 6), "A│ HANDLES (authority) · tile col links to B · ★ = ambiguous", f11, LAYER_A)
    hy = y + 24
    for h in handles:
        star = "★" if h.get("ambiguous") else " "
        tile = h.get("tile") or "  "
        said = f" | SAID:{truncate(str(h.get('said')), 24)}" if h.get("said") else ""
        line = (
            f"{star} {tile:>2}  beat {h.get('beat'):>4}  t={float(h.get('t') or 0):6.1f}  "
            f"{(h.get('kind') or '?'):12}  e={float(h.get('energy') or 0):6.1f}  "
            f"{truncate(h.get('kind_why') or '', 36)}{said}"
        )
        text((pad + 12, hy), truncate(line, 145), f10, WARN if h.get("ambiguous") else INK)
        hy += 12
        if hy > y + bot_h - 28:
            break

    n_amb = len(doc.get("disambiguate") or [])
    text(
        (pad + 12, height - pad - 14),
        truncate(
            f"hybrid {SCHEMA} · disambiguate={n_amb} · "
            f"read page.json first · B is SEEN/shape only · glance for ★ tiles",
            140,
        ),
        f10,
        DIM,
    )
    return img


def write_protocol_md(doc: dict[str, Any]) -> str:
    f = doc["facts"]
    g = f.get("grammar") or {}
    c = f.get("coupling") or {}
    lines = [
        "# Hybrid residual page",
        "",
        "## Read order",
        "1. **Layer A** — `page.json` (facts, tiles, handles) — **authority**",
        "2. **Layer B** — `page.png` (timeline shape + SEEN tiles) — **appearance**",
        "",
        "## Rules",
    ]
    for r in doc["protocol"]["rules"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "## Facts (from A — do not re-OCR)",
        f"- **title:** {f.get('title')}",
        f"- **grammar:** `{g.get('label')}` conf={g.get('confidence')}",
        f"- **mode:** `{c.get('mode')}`",
        f"- **beats:** {f.get('n_beats')} · interval={f.get('interval')}s · mean_e={f.get('mean_energy')}",
        f"- **kinds:** {f.get('kind_counts')}",
        f"- **cuts_total:** {c.get('cuts_total')} · cut_density_in_speech={c.get('cut_density_in_speech')}",
        f"- **WORDS trust:** {f.get('words_trust')}",
        "",
        "## SEEN tiles (open B; metadata in A)",
    ]
    for t in doc["tiles"]:
        amb = " ★" if t.get("ambiguous") else ""
        lines.append(
            f"- **{t['id']}**{amb} beat {t.get('beat')} @{t.get('clock')} "
            f"`{t.get('kind')}` e={t.get('energy')} conf={t.get('kind_confidence')}"
            + (f" alt=`{t.get('kind_alt')}`" if t.get("kind_alt") else "")
            + f" → `{t.get('resolve')}`"
        )
    if doc.get("disambiguate"):
        lines += ["", "## Disambiguate first", ""]
        for d in doc["disambiguate"]:
            lines.append(
                f"- {d['tile']} beat {d['beat']}: {d['kind']} vs {d.get('kind_alt')} "
                f"(conf={d.get('conf')}) → `{d['resolve']}`"
            )
    lines += [
        "",
        "## Better than sum",
    ]
    for b in doc["protocol"]["better_than_sum"]:
        lines.append(f"- {b}")
    lines.append("")
    return "\n".join(lines)


def write_hybrid(
    workdir: Path,
    tiles: int = 8,
    width: int = 1600,
    height: int = 1200,
    out_prefix: str | None = None,
) -> dict[str, Path]:
    workdir = Path(workdir)
    doc = build_layer_a(workdir, tiles_n=tiles)
    img = render_layer_b(workdir, doc, width=width, height=height)

    prefix = out_prefix or "page"
    path_png = workdir / f"{prefix}.png"
    path_json = workdir / f"{prefix}.json"
    path_md = workdir / f"{prefix}.md"

    img.save(path_png, "PNG", optimize=True)
    path_json.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    path_md.write_text(write_protocol_md(doc), encoding="utf-8")

    return {"png": path_png, "json": path_json, "md": path_md, "doc": doc}  # type: ignore[dict-item]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Hybrid residual page: Layer A JSON + Layer B image"
    )
    p.add_argument("workdir", type=Path)
    p.add_argument("--tiles", type=int, default=8)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1200)
    p.add_argument("--prefix", default="page", help="output basename (default: page)")
    args = p.parse_args(argv)

    paths = write_hybrid(
        args.workdir,
        tiles=args.tiles,
        width=args.width,
        height=args.height,
        out_prefix=args.prefix,
    )
    doc = paths["doc"]
    print(
        f"hybrid {SCHEMA} → {paths['json'].name} + {paths['png'].name} + {paths['md'].name}"
    )
    print(
        f"  facts: grammar={doc['facts']['grammar'].get('label')} "
        f"tiles={len(doc['tiles'])} disambiguate={len(doc['disambiguate'])} "
        f"timeline_samples={doc['timeline']['n_samples']}"
    )
    print(f"  {paths['png']} ({paths['png'].stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
