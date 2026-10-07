"""REINFORCE in one picture: reward minus a running-average baseline, then push every answer by that.

Render: tools/animations/render.sh <name>
"""

import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: E402

# One answer to each of four different prompts. Baseline 0.40 -> advantages +0.6 / -0.4.
ROWS = [
    ("23 * 4 + 17", "92+17 = 109", "109", 1),
    ("58 * 3 + 41", "174+41 = 225", "225", 0),
    ("47 * 6 + 35", "282+35 = 317", "317", 1),
    ("91 * 7 + 12", "647+12 = 659", "659", 0),
]
BASELINE = 0.40
ADV = [r - BASELINE for *_, r in ROWS]


class REINFORCE(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = T("REINFORCE: reward minus a running baseline", 32, weight=BOLD).to_edge(UP, buff=0.35)
        ys = [1.0, 0.05, -0.9, -1.85]
        x_ans, x_rew, x_adv, x_prob = -3.55, 1.0, 2.8, 5.0
        heads = VGroup(
            T("one answer per prompt", 20, DIM).move_to([x_ans, 1.8, 0]),
            T("reward", 20, DIM).move_to([x_rew, 1.8, 0]),
            T("advantage", 20, DIM).move_to([x_adv, 1.8, 0]),
            T("probability", 20, DIM).move_to([x_prob, 1.8, 0]),
        )

        # baseline panel: the average reward of past batches
        base_label = T("running baseline", 20, ACCENT)
        base_val = T(f"{BASELINE:.2f}", 30, ACCENT, weight=BOLD)
        base = VGroup(base_label, base_val).arrange(RIGHT, buff=0.25).next_to(title, DOWN, buff=0.3)
        base_note = T("(average reward of earlier batches)", 17, DIM).next_to(base, RIGHT, buff=0.25)

        # 1. sample one answer for each prompt in the batch
        self.play(FadeIn(title), run_time=0.6)
        rows = VGroup()
        for (q, reason, ans, _), y in zip(ROWS, ys):
            rows.add(answer_box(f"{q}?  {reason}", ans, size=16).move_to([x_ans, y, 0]))
        self.play(FadeIn(heads[0]), LaggedStart(*[FadeIn(r, shift=0.3 * RIGHT) for r in rows], lag_ratio=0.25),
                  run_time=1.6)

        # 2. score them
        rewards = VGroup(*[T(str(r), 30, GOOD if r else BAD, weight=BOLD).move_to([x_rew, y, 0])
                           for (*_, r), y in zip(ROWS, ys)])
        self.play(FadeIn(heads[1]), run_time=0.3)
        for (*_, r), row, rew in zip(ROWS, rows, rewards):
            self.play(row[0].animate.set_stroke(GOOD if r else BAD, 2.5), FadeIn(rew, scale=1.4), run_time=0.3)
        self.wait(0.3)

        # 3. compare each reward to the running baseline
        self.play(FadeIn(base, shift=0.2 * DOWN), FadeIn(base_note), run_time=0.7)
        formula = T("advantage = reward − baseline", 22, ACCENT, font=MONO).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(heads[2]), FadeIn(formula), run_time=0.5)
        advs = VGroup(*[T(f"{a:+.1f}", 30, GOOD if a > 0 else BAD, weight=BOLD).move_to([x_adv, y, 0])
                        for a, y in zip(ADV, ys)])
        self.play(LaggedStart(*[TransformFromCopy(r, a) for r, a in zip(rewards, advs)], lag_ratio=0.2),
                  run_time=1.3)
        self.wait(0.6)

        # 4. push every token of each answer by its advantage
        w0, h = 1.0, 0.36
        bars = VGroup(*[Rectangle(width=w0, height=h, stroke_width=0, fill_color=BLUE, fill_opacity=0.85)
                        .move_to([x_prob - 0.6, y, 0], aligned_edge=LEFT) for y in ys])
        self.play(FadeIn(heads[3]), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1),
                  run_time=0.8)
        targets = VGroup(*[
            Rectangle(width=w0 + 0.9 * a, height=h, stroke_width=0,
                      fill_color=GOOD if a > 0 else BAD, fill_opacity=0.9)
            .move_to([x_prob - 0.6, y, 0], aligned_edge=LEFT) for a, y in zip(ADV, ys)])
        arrows = VGroup(*[T("▲" if a > 0 else "▼", 22, GOOD if a > 0 else BAD).next_to(t, RIGHT, buff=0.12)
                          for a, t in zip(ADV, targets)])
        self.play(*[Transform(b, t) for b, t in zip(bars, targets)], FadeIn(arrows), run_time=1.3)
        self.wait(0.4)

        # 5. the baseline drifts toward this batch's average reward (0.5)
        update = T("baseline ← 0.9 · baseline + 0.1 · batch mean (0.5)", 22, ACCENT, font=MONO).move_to(formula)
        self.play(FadeOut(formula, shift=0.2 * DOWN), FadeIn(update, shift=0.2 * UP), run_time=0.5)
        new_val = T(f"{0.9 * BASELINE + 0.1 * 0.5:.2f}", 30, ACCENT, weight=BOLD).move_to(base_val)
        self.play(Transform(base_val, new_val), run_time=0.6)
        self.play(Indicate(base_val, color=ACCENT), run_time=0.6)
        self.wait(0.6)

        takeaway = T("Beat the recent average → more likely.  Fall short → less likely.", 22).move_to(update)
        self.play(FadeOut(update, shift=0.2 * DOWN), FadeIn(takeaway, shift=0.2 * UP), run_time=0.6)
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)
