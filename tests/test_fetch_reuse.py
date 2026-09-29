"""fetch() may only reuse or adopt a *finished* download.

A split-stream yt-dlp download (bestvideo+bestaudio) leaves ``source.f136.mp4``,
``source.f251.webm`` and ``source.temp.mp4`` behind when interrupted or when the
merge fails. Adopting one of those on a re-run would ingest a video-only or
audio-only file and never call yt-dlp again.
"""
from pathlib import Path

from video_perceive.perceive import _finished_video


def test_finished_names_are_adopted(tmp_path: Path):
    for name in ("source.mp4", "source.webm", "source.mkv", "source.mov", "clip.mp4"):
        (tmp_path / name).write_bytes(b"x")
        assert _finished_video(tmp_path / name), name


def test_intermediates_are_not_adopted(tmp_path: Path):
    for name in ("source.f136.mp4", "source.f251.webm", "source.temp.mp4",
                 "source.mp4.part", "source.F18.mp4", "source.en.srt"):
        (tmp_path / name).write_bytes(b"x")
        assert not _finished_video(tmp_path / name), name


def test_directory_named_like_a_video_is_not_adopted(tmp_path: Path):
    (tmp_path / "source.mp4").mkdir()
    assert not _finished_video(tmp_path / "source.mp4")
