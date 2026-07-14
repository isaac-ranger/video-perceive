"""Minimal motion→ASCII helper (vendored; optional visual maps only)."""
from __future__ import annotations

from PIL import Image

ASCII_MOTION = " .:-=+*#@█"


def to_ascii(img: Image.Image, width: int = 70, chars: str = ASCII_MOTION) -> str:
    """Convert a (typically diff-boosted) grayscale/RGB PIL image to ASCII."""
    if img.mode != "L":
        img = img.convert("L")
    aspect = img.height / max(img.width, 1)
    new_height = max(1, int(width * aspect * 0.45))
    img = img.resize((width, new_height))
    pixels = list(img.get_flattened_data()) if hasattr(img, "get_flattened_data") else list(img.getdata())
    out: list[str] = []
    for i, pixel in enumerate(pixels):
        out.append(chars[pixel * len(chars) // 256])
        if (i + 1) % width == 0:
            out.append("\n")
    return "".join(out)
