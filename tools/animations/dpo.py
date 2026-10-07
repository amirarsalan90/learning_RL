"""DPO in one picture: raise the chosen answer, lower the rejected one, both measured against a frozen reference.

Render: tools/animations/render.sh <name>
"""

import math
import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: E402

BETA = 0.1
CHOSEN_END, REJECTED_END = 4.0, -6.0  # log π − log π_ref after training, in nats
SCALE = 0.25  # plot units per nat


def loss(margin):
    return -math.log(1 / (1 + math.exp(-margin)))


class DPO(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = T("DPO: prefer the chosen answer over the rejected one", 32, weight=BOLD).to_edge(UP, buff=0.35)
        prompt = VGroup(T("prompt", 20, DIM), T(QUESTION, 24, font=MONO)).arrange(RIGHT, buff=0.3)
        prompt.next_to(title, DOWN, buff=0.3)

        # 1. one pair from a fixed dataset, built once before training
        x_ans = -3.3
        chosen = answer_box("47*6 = 282, 282+35 = 317", "317", width=6.0)
        rejected = answer_box("47*6 = 272, 272+35 = 307", "307", width=6.0)
        chosen[0].set_stroke(GOOD, 2.5)
        rejected[0].set_stroke(BAD, 2.5)
        chosen.move_to([x_ans, 0.75, 0])
        rejected.move_to([x_ans, -0.85, 0])
        c_lab = T("chosen", 20, GOOD).next_to(chosen, UP, buff=0.12).align_to(chosen, LEFT)
        r_lab = T("rejected", 20, BAD).next_to(rejected, UP, buff=0.12).align_to(rejected, LEFT)
        note = T("a fixed dataset of pairs, made once before training", 18, DIM).next_to(rejected, DOWN, buff=0.3)

        self.play(FadeIn(title), FadeIn(prompt), run_time=0.7)
        self.play(FadeIn(c_lab), FadeIn(chosen, shift=0.3 * RIGHT), run_time=0.6)
        self.play(FadeIn(r_lab), FadeIn(rejected, shift=0.3 * RIGHT), run_time=0.6)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.4)

        # 2. how much more (or less) likely the model makes each answer than the frozen reference does
        x_c, x_r, y_zero, bw = 3.0, 4.7, 0.0, 0.9
        head = T("log π − log π_ref", 20, DIM, font=MONO).move_to([(x_c + x_r) / 2, 2.0, 0])
        zero = Line([x_c - 0.8, y_zero, 0], [x_r + 0.8, y_zero, 0], stroke_color=ACCENT, stroke_width=3)
        zero_lab = T("frozen\nreference", 17, ACCENT).next_to(zero, LEFT, buff=0.15)
        heads = VGroup(T("chosen", 18, GOOD).move_to([x_c, 1.62, 0]), T("rejected", 18, BAD).move_to([x_r, 1.62, 0]))

        t = ValueTracker(0.0)

        def bar(x, end, color):
            def make():
                h = t.get_value() * end * SCALE
                r = Rectangle(width=bw, height=max(abs(h), 0.001), stroke_width=0, fill_color=color, fill_opacity=0.9)
                return r.move_to([x, y_zero, 0], aligned_edge=DOWN if h >= 0 else UP)
            return always_redraw(make)

        bars = VGroup(bar(x_c, CHOSEN_END, GOOD), bar(x_r, REJECTED_END, BAD))
        self.play(FadeIn(head), Create(zero), FadeIn(zero_lab), FadeIn(heads), run_time=0.8)
        self.add(bars)

        def readout():
            c, r = t.get_value() * CHOSEN_END, t.get_value() * REJECTED_END
            m = BETA * (c - r)
            g = VGroup(
                T(f"margin = β · (chosen − rejected) = {m:.2f}", 20, FG, font=MONO),
                T(f"loss = −log σ(margin) = {loss(m):.2f}", 20, ACCENT, font=MONO),
            ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
            return g.to_edge(DOWN, buff=0.35)

        numbers = always_redraw(readout)
        self.play(FadeIn(numbers), FadeOut(note), run_time=0.5)
        self.wait(0.6)

        # 3. training pushes the chosen answer up and the rejected one down until the margin is comfortable
        arrows = VGroup(T("▲", 24, GOOD).move_to([x_c, y_zero + CHOSEN_END * SCALE + 0.3, 0]),
                        T("▼", 24, BAD).move_to([x_r, y_zero + REJECTED_END * SCALE - 0.3, 0]))
        self.play(t.animate.set_value(1.0), run_time=3.2, rate_func=smooth)
        self.play(FadeIn(arrows), run_time=0.4)
        self.wait(1.0)

        numbers.clear_updaters()
        takeaway = T("No reward model, no sampling: just pairs and a frozen reference.", 22).to_edge(DOWN, buff=0.45)
        self.play(FadeOut(numbers, shift=0.2 * DOWN), FadeIn(takeaway, shift=0.2 * UP), run_time=0.6)
        self.wait(2.4)
        for b in bars:
            b.clear_updaters()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)
