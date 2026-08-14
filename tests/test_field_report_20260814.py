"""Tests for the 2026-08-14 field-report fixes.

The specimen: a 22-minute tinkerer screencast (79txDFqcrpU) whose ingest died
four ways before it lived. Each failure bought a rule:

- F1: yt-dlp's stderr vanished into a CalledProcessError that printed only the
  command — `run()` now surfaces the stderr tail before re-raising.
- F4: the dead fetch left an srt-only workdir with no marker; reader verbs
  would have walked it as whole — ingest now drops a sentinel, readers refuse.
- F6: auto-sub rolling windows served every line ~3x; 91.6k chars of cue text
  held 19.8k of speech (the .srt file itself was 38.8k) — `transcript` folds
  them by tail/head overlap, guarded so a coincidental boundary match on
  non-rolling captions cannot manufacture or eat words (QA finding, same day).
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


def test_fold_never_manufactures_words():
    # QA specimens: 1-char coincidental overlaps on NON-rolling captions used
    # to splice mid-word ("I see veryone knows"). A guarded fold concatenates.
    assert fold_rolling_captions(["I see", "everyone knows"]) == (
        "I see everyone knows"
    )
    assert fold_rolling_captions(["go", "oh no"]) == "go oh no"


def test_fold_preserves_genuine_repetition():
    # "no" legitimately said twice across a cue boundary is speech, not a
    # rolling window — a 2-char overlap must not eat it.
    assert fold_rolling_captions(["the answer is no", "no means no"]) == (
        "the answer is no no means no"
    )


def test_fold_rejects_non_boundary_overlap():
    # ≥5 chars but splicing mid-word in the incoming cue: still not a fold.
    assert fold_rolling_captions(["I like corner", "cornered animals run"]) == (
        "I like corner cornered animals run"
    )


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


# --- --no-words honesty (QA finding 3) ---------------------------------------

def test_words_reach_skipped_does_not_claim_a_read():
    from video_perceive.perceive import words_reach

    blank = [{"beat": i, "said": "", "shown": ""} for i in range(5)]
    r = words_reach(blank, words_source="skipped")
    assert r["status"] == "empty_by_request"
    assert "deliberately not read" in r["note"]
    # the two false sentences the old stamp emitted must both be gone
    assert "was read" not in r["note"]
    assert "speechless" not in r["note"]


# --- sentinel lifecycle, end to end (kills mutations M4/M8) -------------------

def _tiny_clip(tmp_path):
    import shutil as _shutil
    import subprocess as _sp

    if not _shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    clip = tmp_path / "clip.mp4"
    _sp.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=192x108:rate=8",
            str(clip),
        ],
        check=True,
    )
    return clip


def _perceive_args(source, workdir):
    import argparse

    return argparse.Namespace(
        source=str(source), workdir=str(workdir), interval=0.5, title=None,
        window=5.0, detail=False, ocr=False, no_words=True, no_ascii=True,
        no_strips=True, force_frames=False, page=False,
    )


def test_ingest_clears_sentinel_on_success(tmp_path):
    from video_perceive.perceive import cmd_perceive

    clip = _tiny_clip(tmp_path)
    workdir = tmp_path / "out"
    cmd_perceive(_perceive_args(clip, workdir))
    assert not (workdir / INGEST_SENTINEL).exists()
    assert (workdir / "summary.json").exists()


def test_ingest_death_leaves_sentinel(tmp_path, monkeypatch):
    import video_perceive.perceive as vp

    clip = _tiny_clip(tmp_path)
    workdir = tmp_path / "out"

    def die(*a, **kw):
        raise RuntimeError("simulated mid-ingest death")

    monkeypatch.setattr(vp, "build_channels", die)
    with pytest.raises(RuntimeError):
        vp.cmd_perceive(_perceive_args(clip, workdir))
    assert (workdir / INGEST_SENTINEL).exists()


def test_typoed_local_path_cannot_lock_a_healthy_residue(tmp_path):
    # QA finding 1: the sentinel used to be written before the source
    # existence check, so `vp ./typo.mp4 --workdir ./healthy` stranded the
    # marker and locked an intact residue behind it.
    from video_perceive.perceive import cmd_perceive

    clip = _tiny_clip(tmp_path)
    workdir = tmp_path / "out"
    cmd_perceive(_perceive_args(clip, workdir))
    assert not (workdir / INGEST_SENTINEL).exists()

    with pytest.raises(SystemExit):
        cmd_perceive(_perceive_args(tmp_path / "typo.mp4", workdir))
    assert not (workdir / INGEST_SENTINEL).exists()
    check_ingest_complete(workdir)  # still readable


def test_every_reader_verb_refuses_a_partial_dir(tmp_path):
    # Dispatch-level, via main() — kills the M8 class (a gate dropped from
    # one verb's dispatch is invisible to unit tests of the gate function).
    import video_perceive.perceive as vp

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / INGEST_SENTINEL).write_text("ingest started\n")

    verbs = [
        ["walk", str(partial)],
        ["summary", str(partial)],
        ["rescore", str(partial)],
        ["transcript", str(partial)],
        ["seen", str(partial)],
        ["strip", str(partial), "--beat", "1"],
        ["glance", str(partial), "--around", "1"],
        ["see", str(partial), "--beat", "1", "--note", "x"],
        ["contact", str(partial)],
        ["diagram", str(partial)],
    ]
    import sys as _sys

    for argv in verbs:
        _sys.argv = ["perceive.py", *argv]
        with pytest.raises(SystemExit) as exc:
            vp.main()
        assert "PARTIAL" in str(exc.value), f"verb {argv[0]} read a partial dir"
