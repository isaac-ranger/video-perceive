"""Tests for content_span_from_colour — which minutes of a file are the work.

Regression anchor: Isaac's La Jetee residue is a Spanish broadcast containing
the film. ~18% of its beats are advert and studio host. The first version of
this detector trimmed only 15s of it, because the opening car advert has a
monochrome dip and an unsmoothed scan stops at the first discordant sample.
"""

import pytest

pytest.importorskip("numpy")

from video_perceive.perceive import content_span_from_colour


def _flags(spec, step=1.0):
    """spec: list of (seconds, is_body) -> boolean sample array."""
    out = []
    for secs, is_body in spec:
        out += [is_body] * int(secs / step)
    return out


def test_clean_file_trims_nothing():
    r = content_span_from_colour(_flags([(300, True)]), 1.0, False)
    assert r["trimmed_head_s"] == 0.0
    assert r["trimmed_tail_s"] == 0.0
    assert r["discordant_segments"] == []
    assert "does not rule out" in r["note"]


def test_head_and_tail_advert_are_trimmed():
    body = _flags([(60, False), (600, True), (60, False)])
    r = content_span_from_colour(body, 1.0, False)
    assert r["start_clock"] == "1:00"
    assert r["trimmed_head_s"] == pytest.approx(60, abs=6)
    assert r["trimmed_tail_s"] == pytest.approx(60, abs=6)
    assert {d["where"] for d in r["discordant_segments"]} == {"head", "tail"}


def test_stray_sample_inside_the_head_does_not_stop_the_trim():
    """The defect this function exists for."""
    # advert, a monochrome dip inside it, more advert, then the film
    body = _flags([(15, False), (15, True), (120, False), (1500, True), (60, False)])
    r = content_span_from_colour(body, 1.0, False)
    assert r["start_s"] > 100, "the 15s dip must not be mistaken for the film"
    assert r["trimmed_head_s"] > 100


def test_naive_first_match_would_have_failed_the_same_input():
    """Pins WHY smoothing is here, so nobody simplifies it back out."""
    body = _flags([(15, False), (15, True), (120, False), (1500, True), (60, False)])
    naive_head = next(i for i, v in enumerate(body) if v)  # first-match scan
    assert naive_head == 15
    assert content_span_from_colour(body, 1.0, False)["start_s"] > naive_head


def test_short_discordant_blips_are_not_reported_as_segments():
    body = _flags([(600, True), (2, False), (600, True)])
    r = content_span_from_colour(body, 1.0, False)
    assert r["discordant_segments"] == []


def test_interior_segment_is_labelled_interior_not_dropped():
    body = _flags([(300, True), (40, False), (300, True)])
    r = content_span_from_colour(body, 1.0, False)
    interiors = [d for d in r["discordant_segments"] if d["where"] == "interior"]
    assert len(interiors) == 1
    assert "may well be part of the film" in r["note"]


def test_empty_input_measures_nothing_rather_than_claiming_a_span():
    r = content_span_from_colour([], 1.0, False)
    assert r["start_s"] == 0.0 and r["end_s"] == 0.0
    assert "nothing measured" in r["note"]


def test_all_discordant_does_not_crash_and_spans_the_file():
    r = content_span_from_colour(_flags([(100, False)]), 1.0, False)
    assert r["end_s"] >= r["start_s"]


def test_basis_states_the_sampling_floor():
    r = content_span_from_colour(_flags([(100, True)]), 2.5, True)
    assert "2.50s" in r["basis"]
    assert "colour" in r["basis"]
