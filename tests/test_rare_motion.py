"""Tests for walk --rare-motion (motion-in-stillness ranking)."""

from video_perceive.perceive import rare_motion_beats


def test_prefers_local_move_in_hold_sea():
    # Mostly HOLD; one LOCAL_MOVE in the middle; one JUMP_CUT ignored by default kinds
    motion = []
    for i in range(30):
        if i == 15:
            motion.append({"beat": i, "kind": "LOCAL_MOVE", "energy": 5.0})
        elif i == 5:
            motion.append({"beat": i, "kind": "JUMP_CUT", "energy": 100.0})
        else:
            motion.append({"beat": i, "kind": "HOLD", "energy": 0.2})
    beats = rare_motion_beats(motion, max_n=3)
    assert 15 in beats
    assert 5 not in beats  # JUMP_CUT not in default kinds


def test_empty():
    assert rare_motion_beats([]) == []


def test_limit_orders_by_time_not_score():
    motion = [
        {"beat": 0, "kind": "HOLD", "energy": 0},
        {"beat": 1, "kind": "LOCAL_MOVE", "energy": 1.0},
        {"beat": 2, "kind": "HOLD", "energy": 0},
        {"beat": 3, "kind": "LOCAL_MOVE", "energy": 50.0},
        {"beat": 4, "kind": "HOLD", "energy": 0},
    ]
    beats = rare_motion_beats(motion, max_n=2, window=2)
    assert beats == sorted(beats)
    assert set(beats) <= {1, 3}
