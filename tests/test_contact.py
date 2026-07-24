"""Tests for contact sheets — the playback surface.

Ordering is the load-bearing property: a contact sheet whose tiles are out of
order is worse than no sheet, because it reads as the film. The t1000.jpg
lexicographic defect is the specimen (69% of La Jetee's alleged cuts), so the
page-order test crosses that boundary deliberately.
"""

import json

import pytest

from video_perceive.perceive import build_contact_sheets, clock, read_interval

PIL = pytest.importorskip("PIL")


def _residue(tmp_path, n_frames=12, interval=0.5, said=None):
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (32, 18), (i * 7 % 256, 40, 90)).save(frames / f"t{i:03d}.jpg")
    (frames / ".interval").write_text(str(interval))
    if said:
        rows = [
            {"beat": b, "said": t, "shown": ""} for b, t in said.items()
        ]
        (tmp_path / "words.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return tmp_path


def test_clock_formats_minutes_seconds():
    assert clock(0) == "0:00"
    assert clock(65.4) == "1:05"
    assert clock(1285) == "21:25"


def test_read_interval_prefers_extraction_stamp(tmp_path):
    _residue(tmp_path, interval=0.25)
    (tmp_path / "summary.json").write_text(json.dumps({"interval": 9.0}))
    assert read_interval(tmp_path) == 0.25


def test_read_interval_falls_back_to_one_second(tmp_path):
    assert read_interval(tmp_path) == 1.0


def test_builds_pages_and_index(tmp_path):
    _residue(tmp_path, n_frames=12)
    pages = build_contact_sheets(tmp_path, step=1, per_page=5, cols=3)
    assert [p.name for p in pages] == ["page_001.jpg", "page_002.jpg", "page_003.jpg"]
    assert (tmp_path / "contact" / "INDEX.md").exists()


def test_no_frames_returns_empty(tmp_path):
    (tmp_path / "frames").mkdir()
    assert build_contact_sheets(tmp_path) == []


def test_pages_are_numerically_ordered_past_999(tmp_path):
    """t1000.jpg must not sort between t100 and t101."""
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir(parents=True)
    for i in (99, 100, 101, 999, 1000, 1001):
        Image.new("RGB", (16, 9), (0, 0, 0)).save(frames / f"t{i:03d}.jpg")
    (frames / ".interval").write_text("1.0")
    from video_perceive.perceive import numeric_frames

    got = [int(p.stem[1:]) for p in numeric_frames(frames)]
    assert got == sorted(got)
    assert build_contact_sheets(tmp_path, step=1, per_page=10)


def test_index_carries_said_text_and_untrusted_provenance(tmp_path):
    _residue(tmp_path, n_frames=8, said={2: "Ceci est l'histoire d'un homme"})
    build_contact_sheets(tmp_path, step=1, per_page=8)
    idx = (tmp_path / "contact" / "INDEX.md").read_text()
    assert "Ceci est l'histoire" in idx
    assert "untrusted" in idx


def test_index_says_so_when_there_is_no_said_text(tmp_path):
    _residue(tmp_path, n_frames=8)
    build_contact_sheets(tmp_path, step=1, per_page=8)
    idx = (tmp_path / "contact" / "INDEX.md").read_text()
    assert "No SAID text" in idx
    assert "summary.reach.words" in idx


def test_index_deduplicates_captions_held_across_beats(tmp_path):
    _residue(tmp_path, n_frames=8, said={1: "same line", 2: "same line", 3: "same line"})
    build_contact_sheets(tmp_path, step=1, per_page=8)
    idx = (tmp_path / "contact" / "INDEX.md").read_text()
    assert idx.count("same line") == 1


def test_lifted_pages_are_declared_not_faithful(tmp_path):
    _residue(tmp_path, n_frames=6)
    build_contact_sheets(tmp_path, step=1, per_page=6, gamma=2.0)
    idx = (tmp_path / "contact" / "INDEX.md").read_text()
    assert "Levels lifted" in idx
    assert "readable, not faithful" in idx


def test_unlifted_pages_make_no_such_claim(tmp_path):
    _residue(tmp_path, n_frames=6)
    build_contact_sheets(tmp_path, step=1, per_page=6)
    assert "Levels lifted" not in (tmp_path / "contact" / "INDEX.md").read_text()


def test_rebuild_clears_stale_pages(tmp_path):
    _residue(tmp_path, n_frames=12)
    build_contact_sheets(tmp_path, step=1, per_page=2)   # 6 pages
    build_contact_sheets(tmp_path, step=1, per_page=12)  # 1 page
    assert len(list((tmp_path / "contact").glob("page_*.jpg"))) == 1


def test_span_is_honoured(tmp_path):
    _residue(tmp_path, n_frames=20)
    build_contact_sheets(tmp_path, start_beat=5, end_beat=10, step=1, per_page=50)
    idx = (tmp_path / "contact" / "INDEX.md").read_text()
    assert "b5–b9" in idx
