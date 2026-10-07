#!/usr/bin/env bash
# Render every animation and write the GIFs to notebooks/figures/.
#   pip install manim   (needs Cairo/Pango; no LaTeX)
#   tools/animations/render.sh [grpo reinforce ppo dpo]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
media=${MEDIA_DIR:-/tmp/rlcourse-manim}
for name in "${@:-reinforce grpo ppo dpo}"; do
  for n in $name; do
    scene=$(echo "$n" | tr a-z A-Z)
    manim -r 960,540 --fps 15 --media_dir "$media" "$here/$n.py" "$scene"
    "$here/to_gif.sh" "$media/videos/$n/540p15/$scene.mp4" "$root/notebooks/figures/$n.gif" 800
  done
done
