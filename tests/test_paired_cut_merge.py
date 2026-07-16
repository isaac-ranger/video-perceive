"""Tests for dissolve-style paired JUMP_CUT merge."""

from video_perceive.perceive import apply_paired_cut_merge


def test_merges_same_signature_adjacent_cuts():
    motion = [
        {"beat": 0, "kind": "OPEN", "energy": 0, "frame_sim": 1.0},
        {
            "beat": 1,
            "kind": "JUMP_CUT",
            "energy": 104.4,
            "frame_sim": -0.15,
            "kind_why": "cut",
        },
        {
            "beat": 2,
            "kind": "JUMP_CUT",
            "energy": 104.4,
            "frame_sim": -0.15,
            "kind_why": "cut",
        },
        {"beat": 3, "kind": "HOLD", "energy": 0.5, "frame_sim": 0.99},
    ]
    out = apply_paired_cut_merge(motion)
    assert out[1]["kind"] == "JUMP_CUT"
    assert out[1].get("paired_cut") is True
    assert out[2]["kind"] == "STIR"
    assert out[2].get("paired_cut_tail") is True
    assert out[2].get("kind_alt") == "JUMP_CUT"


def test_leaves_dissimilar_cuts():
    motion = [
        {"beat": 1, "kind": "JUMP_CUT", "energy": 100, "frame_sim": 0.1},
        {"beat": 2, "kind": "JUMP_CUT", "energy": 20, "frame_sim": 0.8},
    ]
    out = apply_paired_cut_merge(motion)
    assert out[1]["kind"] == "JUMP_CUT"
    assert out[1].get("paired_cut_tail") is not True


def test_requires_adjacent_beat_ids():
    motion = [
        {"beat": 1, "kind": "JUMP_CUT", "energy": 50, "frame_sim": 0.0},
        {"beat": 5, "kind": "JUMP_CUT", "energy": 50, "frame_sim": 0.0},
    ]
    out = apply_paired_cut_merge(motion)
    assert out[1]["kind"] == "JUMP_CUT"
