"""Shared style for the algorithm animations (dark card so they read on light and dark pages)."""

from manim import BOLD, NORMAL, RIGHT, RoundedRectangle, Text, VGroup

BG = "#161b22"
FG = "#e6edf3"
DIM = "#8b949e"
CARD = "#21262d"
GOOD = "#3fb950"
BAD = "#f85149"
ACCENT = "#e3a33b"  # baselines, critics, references (orange, as in the notebook diagrams)
BLUE = "#58a6ff"

SANS = "Inter"
MONO = "DejaVu Sans Mono"

QUESTION = "What is 47 * 6 + 35?"


def T(s, size=24, color=FG, font=SANS, weight=NORMAL):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


def answer_box(reason, ans, width=6.6, size=17):
    """A sampled completion: its reasoning plus the <answer> tag, in a rounded card."""
    box = RoundedRectangle(corner_radius=0.12, width=width, height=0.72,
                           stroke_color=DIM, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
    line = VGroup(T(reason, size, font=MONO), T(f"<answer>{ans}</answer>", size, BLUE, font=MONO))
    line.arrange(RIGHT, buff=0.2)
    line.scale_to_fit_width(min(line.width, width - 0.4)).move_to(box)
    return VGroup(box, *line)
