"""Tests for the 2026-08-14 field-report fixes.

The specimen: a 22-minute tinkerer screencast (79txDFqcrpU) whose ingest died
four ways before it lived. Each failure bought a rule:

- F1: yt-dlp's stderr vanished into a CalledProcessError that printed only the
  command — `run()` now surfaces the stderr tail before re-raising.
- F4: the dead fetch left an srt-only workdir with no marker; reader verbs
  would have walked it as whole — ingest now drops a sentinel, readers refuse.
- F6: auto-sub rolling windows served every line ~3x; 38.8k chars of cue text
  held 19.7k of speech — `transcript` folds them by tail/head overlap.
- F7: burned-in text was detected AND unreadable at 360p, and nothing said the
  pixel grid was why — reach now carries the source resolution.
"""

import subprocess

import pytest

from video_perceive.perceive import (
    INGEST_SENTINEL,
    build_summary,
    check_ingest_complete,
    fold_rolling_captions,
    run,
    video_resolution,
)


# --- F6: rolling-caption fold -------------------------------------------------

def test_fold_collapses_rolling_window():
    # The YouTube shape: cue N repeats the tail of cue N-1 as its head.
    cues = [
        "As you can see, I'm here in Hermes",
        "As you can see, I'm here in Hermes agent, and this is a terminal",
        "agent, and this is a terminal session on my DGX",
        "session on my DGX Spark where I run this model",
    ]
    folded = fold_rolling_captions(cues)
    assert folded == (
        "As you can see, I'm here in Hermes agent, and this is a terminal "
        "session on my DGX Spark where I run this model"
    )
    # every source word survives exactly once
    assert folded.count("terminal") == 1
    assert folded.count("DGX") == 1


def test_fold_without_overlap_concatenates():
    assert fold_rolling_captions(["one thing.", "another thing."]) == (
        "one thing. another thing."
    )


def test_fold_identical_cues_collapse():
    assert fold_rolling_captions(["same line", "same line", "same line"]) == (
        "same line"
    )


def test_fold_empty_and_whitespace_cues():
    assert fold_rolling_captions([]) == ""
    assert fold_rolling_captions(["", "  ", "word"]) == "word"


# --- F4: ingest sentinel ------------------------------------------------------

def test_partial_ingest_refused(tmp_path):
    (tmp_path / INGEST_SENTINEL).write_text("ingest started\n")
    with pytest.raises(SystemExit) as exc:
        check_ingest_complete(tmp_path)
    assert "PARTIAL" in str(exc.value)


def test_complete_ingest_passes(tmp_path):
    check_ingest_complete(tmp_path)  # no sentinel → no complaint


# --- F1: subprocess stderr surfaced ------------------------------------------

def test_run_surfaces_stderr_tail(capsys):
    with pytest.raises(subprocess.CalledProcessError):
        run(["bash", "-c", "echo the-actual-reason >&2; exit 3"])
    err = capsys.readouterr().err
    assert "the-actual-reason" in err
    assert "exited 3" in err


def test_run_success_prints_nothing(capsys):
    run(["true"])
    assert capsys.readouterr().err == ""


# --- F7: resolution in the reach stamp ---------------------------------------

def test_reach_carries_resolution():
    s = build_summary("t", "src", 1.0, [], [], resolution="640x360")
    assert "640x360" in s["reach"]["resolution"]
    assert "reach limit" in s["reach"]["resolution"]


def test_reach_names_unmeasured_resolution():
    s = build_summary("t", "src", 1.0, [], [])
    assert "unmeasured" in s["reach"]["resolution"]


def test_video_resolution_none_on_garbage(tmp_path):
    not_video = tmp_path / "nope.mp4"
    not_video.write_text("this is not a video")
    assert video_resolution(not_video) is None
