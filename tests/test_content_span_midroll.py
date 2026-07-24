"""A mid-roll break must not truncate the work.

Found by test_interior_segment_is_labelled_interior_not_dropped: the first
implementation took the LONGEST body run as the content span, so a film split
by an advert break reported only its larger half — and reported it with a
clean span and no warning. Silent truncation reads as coverage.
"""

import pytest

pytest.importorskip("numpy")

from video_perceive.perceive import content_span_from_colour


def _flags(spec, step=1.0):
    out = []
    for secs, is_body in spec:
        out += [is_body] * int(secs / step)
    return out


def test_midroll_break_keeps_both_halves():
    body = _flags([(600, True), (60, False), (600, True)])
    r = content_span_from_colour(body, 1.0, False)
    assert r["start_s"] == pytest.approx(0, abs=6)
    assert r["end_s"] == pytest.approx(1260, abs=12), "second half must not be dropped"
    assert r["trimmed_tail_s"] == pytest.approx(0, abs=6)


def test_midroll_break_is_reported_as_interior():
    body = _flags([(600, True), (60, False), (600, True)])
    r = content_span_from_colour(body, 1.0, False)
    interiors = [d for d in r["discordant_segments"] if d["where"] == "interior"]
    assert len(interiors) == 1
    assert interiors[0]["seconds"] == pytest.approx(60, abs=6)


def test_two_midroll_breaks_all_interior():
    body = _flags([(400, True), (40, False), (400, True), (40, False), (400, True)])
    r = content_span_from_colour(body, 1.0, False)
    assert [d["where"] for d in r["discordant_segments"]] == ["interior", "interior"]


def test_tiny_body_blip_in_the_head_still_does_not_anchor_the_span():
    """The mid-roll fix must not undo the smoothing fix."""
    body = _flags([(15, False), (15, True), (120, False), (1500, True), (60, False)])
    r = content_span_from_colour(body, 1.0, False)
    assert r["start_s"] > 100
    assert r["trimmed_tail_s"] > 40


def test_head_advert_and_midroll_together():
    body = _flags([(120, False), (500, True), (50, False), (500, True), (90, False)])
    r = content_span_from_colour(body, 1.0, False)
    where = [d["where"] for d in r["discordant_segments"]]
    assert where == ["head", "interior", "tail"]
    assert r["start_s"] == pytest.approx(120, abs=6)
    assert r["end_s"] == pytest.approx(1170, abs=12)
