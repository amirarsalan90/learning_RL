#!/usr/bin/env bash
# Convert a rendered Manim video to a small looping GIF with an optimised palette.
#   tools/animations/to_gif.sh input.mp4 output.gif [width]
set -euo pipefail
in=$1 out=$2 width=${3:-800}
filters="fps=15,scale=${width}:-1:flags=lanczos"
ffmpeg -loglevel error -y -i "$in" -vf "$filters,palettegen=max_colors=64:stats_mode=diff" /tmp/palette.png
ffmpeg -loglevel error -y -i "$in" -i /tmp/palette.png \
  -lavfi "$filters [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$out"
ls -lh "$out"
