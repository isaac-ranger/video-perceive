#!/usr/bin/env python3
"""Agent watched-video library for video-perceive residues.

Scans the shelf (and optional extra roots) for residual workdirs, writes
CATALOG.json + CATALOG.md. Optional annotations.jsonl holds witness notes,
pre-registered predictions, and findings that outlive a single rescore.

Isaac TUNING open item: glob both */summary.json and */out/summary.json;
skip glances/; carry source, witness, prediction, findings, checksums.

Usage:
  catalog.py rebuild [--root DIR]...
  catalog.py list [--grammar LABEL] [--witness AGENT]
  catalog.py show <id>
  catalog.py note <id> --author AGENT --text "..." [--kind finding|prediction|watch]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

LIBRARY_DIR = Path(__file__).resolve().parent


def _default_shelf() -> Path:
    """Where residues live.

    On the household shelf this module sits beside the workdirs, so its parent
    IS the shelf. Once packaged, the same expression points at the installed
    source tree — a directory with no residues in it, which would make
    `rebuild` quietly catalogue nothing and report success. Resolution order:
    explicit env var, then the package parent only if it actually holds
    residues, then the working directory.
    """
    env = os.environ.get("VIDEO_PERCEIVE_SHELF")
    if env:
        return Path(env).expanduser()
    parent = LIBRARY_DIR.parent
    if any(parent.glob("*/summary.json")) or any(parent.glob("*/out/summary.json")):
        return parent
    return Path.cwd()


DEFAULT_SHELF = _default_shelf()
LIBRARY_STORE = Path(
    os.environ.get("VIDEO_PERCEIVE_LIBRARY") or (DEFAULT_SHELF / "library")
)
CATALOG_JSON = LIBRARY_STORE / "CATALOG.json"
CATALOG_MD = LIBRARY_STORE / "CATALOG.md"
ANNOTATIONS = LIBRARY_STORE / "annotations.jsonl"

# Nested summary that is still the workdir root (Cairn/Pi double-blind miss)
OUT_SUMMARY = re.compile(r"/out/summary\.json$")
GLANCE_PART = "glances"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path, limit: int | None = 2_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit is None:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit))
            st = path.stat()
            h.update(f"\0{st.st_size}\0{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def extract_video_id(source: str | None) -> str | None:
    """YouTube id from URL or bare id. Never scrape local filesystem paths
    (avoids false hits like 'eo-perceive' inside '.../video-perceive/...').
    """
    if not source:
        return None
    s = source.strip()
    # bare youtube-style id
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    if "://" not in s and not s.startswith("/") and "/" not in s:
        # non-11 bare tokens (shorts ids are usually 11; leave others alone)
        return None
    try:
        u = urlparse(s)
    except Exception:
        return None
    host = (u.netloc or "").lower()
    if not host:
        return None  # local path — caller may use workdir name
    if "youtu.be" in host:
        vid = u.path.strip("/").split("/")[0] or None
        return vid if vid and re.fullmatch(r"[A-Za-z0-9_-]{11}", vid) else None
    if "youtube" in host or "youtube-nocookie" in host:
        qs = parse_qs(u.query)
        if "v" in qs and re.fullmatch(r"[A-Za-z0-9_-]{11}", qs["v"][0]):
            return qs["v"][0]
        parts = [p for p in u.path.split("/") if p]
        for key in ("shorts", "embed", "live", "v"):
            if key in parts:
                i = parts.index(key)
                if i + 1 < len(parts) and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[i + 1]):
                    return parts[i + 1]
    return None


def infer_witness(workdir: Path, summary: dict, shelf: Path) -> str | None:
    """Best-effort witness: dir prefix, then seen.jsonl authors, then demo→house."""
    name = workdir.name
    if name.startswith("demo-"):
        return "house"
    for agent in ("grok", "isaac", "cairn", "pi", "builder", "rowan", "sol"):
        if name == agent or name.startswith(agent + "-"):
            return agent
    seen = workdir / "seen.jsonl"
    if seen.exists():
        authors: list[str] = []
        try:
            for line in seen.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                a = (row.get("author") or "").strip().lower()
                if a and a not in ("agent",):
                    authors.append(a)
        except Exception:
            pass
        if authors:
            # most common non-generic author
            return max(set(authors), key=authors.count)
    # ownership by filesystem (best effort)
    try:
        import pwd

        owner = pwd.getpwuid(workdir.stat().st_uid).pw_name
        if owner in ("grok", "isaac", "cairn", "pi", "builder", "rowan", "carl"):
            return owner
    except Exception:
        pass
    return None


def find_summary_paths(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for p in root.rglob("summary.json"):
        parts = set(p.parts)
        if GLANCE_PART in parts:
            continue
        # only residual roots: .../summary.json or .../out/summary.json
        parent = p.parent
        if parent.name == "out":
            # workdir is parent of out
            found.append(p)
        elif (parent / "motion.jsonl").exists() or (parent / "meta.json").exists() or (
            parent / "source.mp4"
        ).exists():
            found.append(p)
        elif p.name == "summary.json" and parent.parent == root:
            found.append(p)
    # de-dupe: prefer workdir-level over nested if both? keep both only if different workdirs
    return sorted(set(found))


def workdir_for_summary(summary_path: Path) -> Path:
    if summary_path.parent.name == "out":
        return summary_path.parent.parent
    return summary_path.parent


def load_annotations() -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    if not ANNOTATIONS.exists():
        return by_id
    for line in ANNOTATIONS.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = row.get("id")
        if not rid:
            continue
        by_id.setdefault(rid, []).append(row)
    return by_id


def entry_id(workdir: Path, video_id: str | None, shelf: Path) -> str:
    try:
        rel = workdir.resolve().relative_to(shelf.resolve())
        return str(rel).replace(os.sep, "/")
    except ValueError:
        if video_id:
            return f"ext:{video_id}:{workdir.name}"
        return f"ext:{workdir.name}"


def build_entry(
    summary_path: Path,
    shelf: Path,
    annotations: dict[str, list[dict]],
) -> dict[str, Any] | None:
    workdir = workdir_for_summary(summary_path)
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as e:
        return {
            "id": entry_id(workdir, None, shelf),
            "error": f"unreadable summary: {e}",
            "path": str(workdir),
            "summary_path": str(summary_path),
        }

    meta: dict = {}
    meta_path = workdir / "meta.json"
    if not meta_path.exists() and (workdir / "out" / "meta.json").exists():
        meta_path = workdir / "out" / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    source = summary.get("source") or meta.get("source")
    title = summary.get("title") or meta.get("title") or workdir.name
    video_id = extract_video_id(source if isinstance(source, str) else None)
    if not video_id:
        # dir naming: agent-VIDEOID or demo-VIDEOID or grok-VIDEOID
        m = re.search(r"(?:^|-)([A-Za-z0-9_-]{11})$", workdir.name)
        if m:
            video_id = m.group(1)
        elif workdir.name.startswith("demo-"):
            rest = workdir.name[5:]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", rest):
                video_id = rest

    grammar = summary.get("grammar") or meta.get("grammar") or {}
    if isinstance(grammar, str):
        grammar = {"label": grammar}
    cut_audit = summary.get("cut_audit") or {}
    coupling = summary.get("coupling") or meta.get("coupling") or {}
    kind_counts = summary.get("kind_counts") or meta.get("kind_counts") or {}
    interval = summary.get("interval") or meta.get("interval")
    n_beats = summary.get("n_beats") or meta.get("n_beats")

    st = summary_path.stat()
    eid = entry_id(workdir, video_id, shelf)
    notes = annotations.get(eid, [])

    seen_authors: list[str] = []
    seen_path = workdir / "seen.jsonl"
    if not seen_path.exists() and (workdir / "out" / "seen.jsonl").exists():
        seen_path = workdir / "out" / "seen.jsonl"
    n_seen = 0
    if seen_path.exists():
        for line in seen_path.read_text().splitlines():
            if not line.strip():
                continue
            n_seen += 1
            try:
                a = json.loads(line).get("author")
                if a:
                    seen_authors.append(str(a))
            except Exception:
                pass

    witness = infer_witness(workdir, summary, shelf)
    # annotations can override / add co-watchers
    co = sorted(
        {
            n.get("author")
            for n in notes
            if n.get("author") and n.get("kind") in (None, "watch", "finding", "prediction")
        }
    )

    source_mp4 = workdir / "source.mp4"
    if not source_mp4.exists():
        source_mp4 = workdir / "out" / "source.mp4"

    return {
        "id": eid,
        "title": title,
        "source": source,
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
        "witness": witness,
        "co_watchers": co,
        "path": str(workdir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "has_source_mp4": source_mp4.exists(),
        "interval": interval,
        "n_beats": n_beats,
        "grammar": {
            "label": grammar.get("label"),
            "fit": grammar.get("fit", grammar.get("confidence")),
            "status": grammar.get("status"),
        },
        "kind_counts": kind_counts,
        "cut_audit": {
            k: cut_audit.get(k)
            for k in (
                "alleged",
                "verified",
                "demoted_world_hold",
                "churn_at_floor",
                "cuts_after_audit",
            )
            if cut_audit
        }
        or None,
        "coupling_mode": coupling.get("mode") if isinstance(coupling, dict) else None,
        "speech_beats": coupling.get("speech_beats") if isinstance(coupling, dict) else None,
        "n_seen_notes": n_seen,
        "seen_authors": sorted(set(seen_authors)),
        "summary_sha256_16": _sha256_file(summary_path),
        "summary_mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "annotations": notes,
        "tags": _tags_for(workdir, grammar, witness),
    }


def _tags_for(workdir: Path, grammar: dict, witness: str | None) -> list[str]:
    tags: list[str] = []
    if workdir.name.startswith("demo-"):
        tags.append("demo")
    if witness and witness != "house":
        tags.append(f"witness:{witness}")
    lab = (grammar or {}).get("label")
    if lab:
        tags.append(f"grammar:{lab}")
    return tags


def rebuild(roots: list[Path], shelf: Path) -> dict:
    annotations = load_annotations()
    entries: list[dict] = []
    seen_paths: set[str] = set()
    for root in roots:
        for sp in find_summary_paths(root):
            key = str(sp.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            ent = build_entry(sp, shelf, annotations)
            if ent:
                entries.append(ent)

    entries.sort(key=lambda e: (e.get("summary_mtime") or "", e.get("id") or ""), reverse=True)

    catalog = {
        "schema": "agent-watched-library/v0",
        "generated_at": _utc_now(),
        "shelf": str(shelf.resolve()),
        "roots": [str(r.resolve()) for r in roots],
        "n_entries": len(entries),
        "entries": entries,
    }
    LIBRARY_STORE.mkdir(parents=True, exist_ok=True)
    CATALOG_JSON.write_text(json.dumps(catalog, indent=2) + "\n")
    CATALOG_MD.write_text(render_markdown(catalog))
    return catalog


def render_markdown(catalog: dict) -> str:
    lines = [
        "# Agent watched library",
        "",
        f"Generated: `{catalog['generated_at']}` · **{catalog['n_entries']}** residues",
        f"Shelf: `{catalog['shelf']}`",
        "",
        "Rebuild: `python3 library/catalog.py rebuild`",
        "",
        "| ID | Witness | Title | Grammar | Beats | Cut audit (V/D/C of A) | Source |",
        "|----|---------|-------|---------|-------|------------------------|--------|",
    ]
    for e in catalog["entries"]:
        if e.get("error"):
            lines.append(f"| `{e.get('id')}` | — | ERROR | — | — | — | {e.get('error')} |")
            continue
        g = (e.get("grammar") or {}).get("label") or "—"
        fit = (e.get("grammar") or {}).get("fit")
        if isinstance(fit, float):
            fit_s = f"{fit:.2f}"
        elif fit is not None:
            fit_s = str(fit)
        else:
            fit_s = None
        gcell = f"{g}" + (f" ({fit_s})" if fit_s is not None else "")
        ca = e.get("cut_audit") or {}
        if ca and ca.get("alleged") is not None:
            audit = f"{ca.get('verified', '—')}/{ca.get('demoted_world_hold', '—')}/{ca.get('churn_at_floor', '—')} of {ca.get('alleged')}"
        else:
            audit = "—"
        title = (e.get("title") or "").replace("|", "\\|")[:40]
        src = e.get("youtube_url") or e.get("source") or "—"
        if isinstance(src, str) and len(src) > 48:
            src = src[:45] + "…"
        src = str(src).replace("|", "\\|")
        lines.append(
            f"| `{e.get('id')}` | {e.get('witness') or '—'} | {title} | {gcell} | {e.get('n_beats') or '—'} | {audit} | {src} |"
        )

    lines += [
        "",
        "## Conventions",
        "",
        "- Workdir on shelf: `{agent}-{youtubeId}` or `demo-{id|slug}`",
        "- Nested `out/summary.json` is included (La Jetée-style)",
        "- `glances/*/summary.json` is **not** a separate watch",
        "- Annotations: `library/annotations.jsonl` via `catalog.py note`",
        "- This catalog is an index of **residues** (what was watched + how it was read), not a media vault",
        "",
    ]
    return "\n".join(lines) + "\n"


def cmd_list(catalog: dict, grammar: str | None, witness: str | None) -> None:
    for e in catalog.get("entries") or []:
        if grammar and (e.get("grammar") or {}).get("label") != grammar:
            continue
        if witness and e.get("witness") != witness:
            continue
        g = (e.get("grammar") or {}).get("label")
        print(f"{e.get('id'):40}  {e.get('witness') or '?':8}  {g or '—':28}  {e.get('title') or ''}")


def cmd_show(catalog: dict, eid: str) -> None:
    for e in catalog.get("entries") or []:
        if e.get("id") == eid or e.get("video_id") == eid or e.get("id", "").endswith(eid):
            print(json.dumps(e, indent=2))
            return
    print(f"not found: {eid}", file=sys.stderr)
    sys.exit(1)


def cmd_note(eid: str, author: str, text: str, kind: str) -> None:
    # resolve id against catalog if present
    if CATALOG_JSON.exists():
        cat = json.loads(CATALOG_JSON.read_text())
        resolved = None
        for e in cat.get("entries") or []:
            if e.get("id") == eid or e.get("video_id") == eid or str(e.get("id", "")).endswith(eid):
                resolved = e["id"]
                break
        if resolved:
            eid = resolved
    row = {
        "id": eid,
        "kind": kind,
        "author": author,
        "text": text,
        "ts": _utc_now(),
    }
    ANNOTATIONS.parent.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"note: {kind} on {eid} by {author}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rebuild = sub.add_parser("rebuild", help="scan shelf and write CATALOG.json + CATALOG.md")
    p_rebuild.add_argument(
        "--root",
        action="append",
        default=[],
        help="extra root to scan (repeatable); default is the video-perceive shelf only",
    )
    p_rebuild.add_argument(
        "--shelf",
        default=str(DEFAULT_SHELF),
        help="primary shelf root (default: tools/video-perceive)",
    )

    p_list = sub.add_parser("list", help="list catalog entries")
    p_list.add_argument("--grammar")
    p_list.add_argument("--witness")

    p_show = sub.add_parser("show", help="show one entry as JSON")
    p_show.add_argument("id")

    p_note = sub.add_parser("note", help="append annotation for an entry")
    p_note.add_argument("id")
    p_note.add_argument("--author", required=True)
    p_note.add_argument("--text", required=True)
    p_note.add_argument(
        "--kind",
        default="finding",
        choices=["finding", "prediction", "watch", "note"],
    )

    args = ap.parse_args()

    if args.cmd == "rebuild":
        shelf = Path(args.shelf)
        roots = [shelf] + [Path(r) for r in args.root]
        cat = rebuild(roots, shelf)
        print(f"catalog: {cat['n_entries']} entries → {CATALOG_JSON} + {CATALOG_MD}")
        return

    if not CATALOG_JSON.exists():
        print("no catalog yet; run: catalog.py rebuild", file=sys.stderr)
        sys.exit(1)
    catalog = json.loads(CATALOG_JSON.read_text())

    if args.cmd == "list":
        cmd_list(catalog, args.grammar, args.witness)
    elif args.cmd == "show":
        cmd_show(catalog, args.id)
    elif args.cmd == "note":
        cmd_note(args.id, args.author, args.text, args.kind)
        # refresh so annotations appear
        shelf = Path(catalog.get("shelf") or DEFAULT_SHELF)
        roots = [Path(r) for r in catalog.get("roots") or [shelf]]
        rebuild(roots, shelf)
        print("catalog rebuilt with annotations")


if __name__ == "__main__":
    main()
