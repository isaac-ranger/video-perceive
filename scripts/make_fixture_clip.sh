#!/usr/bin/env bash
# Generate a tiny synthetic clip (no external media) for smoke tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/fixtures/synthetic"
mkdir -p "$OUT_DIR"
CLIP="$OUT_DIR/clip.mp4"

# 3 seconds: color blocks + a hard cut + motion (drawtext optional)
# Uses lavfi only — no input files.
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x224466:s=320x240:d=1,format=yuv420p" \
  -f lavfi -i "color=c=0xAA3333:s=320x240:d=1,format=yuv420p" \
  -f lavfi -i "color=c=0x33AA66:s=320x240:d=1,format=yuv420p" \
  -filter_complex "[0][1][2]concat=n=3:v=1:a=0,fps=8" \
  -t 3 "$CLIP"

echo "wrote $CLIP"
ls -la "$CLIP"
