"""Grammar: quiet screencasts must not become trajectory on modest centroid drift."""

from video_perceive.perceive import infer_grammar


def test_pedagogical_with_modest_drift_high_quiet():
    """Matt Pocock-style: long HOLD/STIR + one cut, drift ~0.14 from PiP/scroll.

    Carries SAID evidence (speech_beats) — the pedagogical label requires
    speech_frac >= 0.10 since the speech gate (a speechless quiet/spike
    histogram is shared by continuous takes and murmurations).
    """
    kinds = {
        "OPEN": 1,
        "LOCAL_MOVE": 30,
        "STIR": 52,
        "JUMP_CUT": 1,
        "HOLD": 100,
    }
    n = sum(kinds.values())
    g = infer_grammar(
        kinds,
        n,
        mean_energy=5.08,
        centroid_drift_x=-0.141,
        coupling={
            "mode": "speech_illustrates_motion",
            "disagree_score": 0.005,
            "cut_density_in_speech": 0.005,
            "speech_runs_spanning_multi_cut": 0,
            "speech_beats": 60,
        },
    )
    assert g["label"] == "pedagogical_pulse"


def test_pedagogical_requires_speech_evidence():
    """Speech gate: same quiet/spike histogram WITHOUT SAID evidence must not
    claim lecture genre (starling misread — rhythm-true, genre-false)."""
    kinds = {
        "OPEN": 1,
        "LOCAL_MOVE": 30,
        "STIR": 52,
        "JUMP_CUT": 1,
        "HOLD": 100,
    }
    n = sum(kinds.values())
    g = infer_grammar(
        kinds,
        n,
        mean_energy=5.08,
        centroid_drift_x=-0.141,
        coupling={"mode": "motion_only", "speech_beats": 0},
    )
    assert g["label"] != "pedagogical_pulse"
    assert g.get("best_guess") != "pedagogical_pulse"


def test_pedagogical_still_requires_quiet_body():
    kinds = {"JUMP_CUT": 20, "LOCAL_MOVE": 5, "HOLD": 5, "STIR": 5}
    n = sum(kinds.values())
    g = infer_grammar(kinds, n, mean_energy=20.0, centroid_drift_x=0.0)
    assert g["label"] != "pedagogical_pulse"


def test_trajectory_when_quiet_borderline_and_high_drift():
    """Borderline quiet + large drift → travel/scene, not lecture."""
    kinds = {
        "OPEN": 1,
        "LOCAL_MOVE": 12,
        "STIR": 8,
        "HOLD": 10,  # quiet_r = 18/31 ≈ 0.58 < 0.60
        "JUMP_CUT": 0,
    }
    n = sum(kinds.values())
    g = infer_grammar(
        kinds,
        n,
        mean_energy=8.0,
        centroid_drift_x=0.25,
        coupling={"mode": "motion_only"},
    )
    assert g["label"] == "trajectory_or_scene"
