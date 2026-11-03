#!/usr/bin/env bash
# MIT OCW: David Jerison, Single Variable Calculus, Lecture 1.
# Media is CC BY-NC-SA 4.0; it is not part of the application license.
set -euo pipefail
example_root="${1:-experiments/real-reproduced}"
mkdir -p "$example_root"
if [ ! -f "$example_root/input.mp4" ]; then
  ffmpeg -nostdin -v error -ss 180 \
    -i 'https://archive.org/download/MIT18.01JF07/ocw-18.01-f07-lec01_300k.mp4' \
    -t 45 -c:v libx264 -preset fast -c:a aac "$example_root/input.mp4"
fi
lecture2doc process "$example_root/input.mp4" --out "$example_root/record" \
  --roi 0.1,0,0.85,0.74 --board blackboard --make-playback
# Optional first-time model download; after this, omit --allow-download.
# lecture2doc process "$example_root/input.mp4" --out "$example_root/record" \
#   --roi 0.1,0,0.85,0.74 --board blackboard --model tiny --language en --allow-download
python scripts/evaluate.py --record "$example_root/record"
printf 'View: lecture2doc serve --record %s/record\n' "$example_root"
