"""Import and CLI smoke (no ffmpeg required)."""
from __future__ import annotations

import video_perceive
from video_perceive import ascii_motion, diagram


def test_version():
    assert video_perceive.__version__


def test_to_ascii_roundtrip():
    from PIL import Image

    img = Image.new("L", (40, 20), color=128)
    s = ascii_motion.to_ascii(img, width=20)
    assert isinstance(s, str)
    assert len(s) > 10


def test_kind_colors_present():
    assert "JUMP_CUT" in diagram.KIND_COLOR
    assert diagram.SCHEMA.startswith("video-perceive")
