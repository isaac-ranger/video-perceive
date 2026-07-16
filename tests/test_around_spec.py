"""Unit tests for glance --around time/beat parsing."""

from video_perceive.perceive import parse_around_spec, resolve_around_beat


def test_parse_beat():
    assert parse_around_spec("42") == ("beat", 42)


def test_parse_seconds_suffix():
    kind, val = parse_around_spec("1285s")
    assert kind == "time"
    assert val == 1285.0


def test_parse_mmss():
    kind, val = parse_around_spec("21:25")
    assert kind == "time"
    assert val == 21 * 60 + 25


def test_parse_hmss():
    kind, val = parse_around_spec("1:02:03")
    assert kind == "time"
    assert val == 3600 + 120 + 3


def test_parse_compound():
    kind, val = parse_around_spec("21m25s")
    assert kind == "time"
    assert val == 21 * 60 + 25


def test_resolve_nearest_beat():
    motion = [
        {"beat": 0, "t": 0.0},
        {"beat": 10, "t": 5.0},
        {"beat": 20, "t": 10.0},
    ]
    beat, note = resolve_around_beat(motion, "9.2s")
    assert beat == 20
    assert "nearest beat 20" in note


def test_resolve_beat_legacy():
    motion = [{"beat": 7, "t": 3.5}]
    beat, note = resolve_around_beat(motion, "7")
    assert beat == 7
    assert note == "beat 7"
