"""Tests for reach.words — an empty channel must say WHY it is empty.

The specimen: a 3661-beat residue of a narrated film whose WORDS channel was
blank because no caption track existed. speech_frac 0.0 removed every
speech-bearing grammar from the option set, and the read still returned a
label with confidence. These assert the distinction is now on the record.
"""

from video_perceive.perceive import words_reach


def _blank(n=10):
    return [{"beat": i, "said": "", "shown": ""} for i in range(n)]


def test_absent_when_no_channel():
    r = words_reach([])
    assert r["status"] == "absent"
    assert r["carried"] is None


def test_present_counts_said_and_shown():
    words = _blank(5)
    words[1]["said"] = "hello"
    words[3]["shown"] = "TITLE CARD"
    r = words_reach(words, words_source="captions")
    assert r["status"] == "present"
    assert r["carried"] == 2
    assert r["note"] is None


def test_blank_with_no_source_is_not_evidence_of_silence():
    r = words_reach(_blank(), words_source="none")
    assert r["status"] == "empty_no_source"
    assert r["carried"] == 0
    assert "NOT evidence" in r["note"]
    assert "--ocr" in r["note"]


def test_blank_with_a_source_read_is_evidence_of_silence():
    r = words_reach(_blank(), words_source="captions")
    assert r["status"] == "empty_source_present"
    assert "speechless" in r["note"]


def test_unknown_provenance_refuses_to_guess():
    r = words_reach(_blank())
    assert r["status"] == "empty_undetermined"
    assert "indistinguishable" in r["note"]


def test_every_empty_status_names_the_reduced_option_set():
    # the point of the stamp: downstream must know the grammar was picked blind
    for src in (None, "none", "captions"):
        r = words_reach(_blank(), words_source=src)
        assert "unreachable" in r["note"]
        assert "montage_over_speech" in r["note"]


def test_present_never_carries_the_blind_warning():
    words = _blank(3)
    words[0]["said"] = "a"
    assert words_reach(words, words_source="captions")["note"] is None
