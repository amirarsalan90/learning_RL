"""PPO in one picture: a critic predicts the outcome after every token, so each token gets its own credit.

Render: tools/animations/render.sh <name>
"""

import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa: E402

TOKENS = ["47*6", "=", "272", ",", "272+35", "=", "307", "<answer>307</answer>"]
# Critic's predicted chance of a correct final answer, before each token; the last point is the real reward.
VALUES = [0.60, 0.62, 0.62, 0.12, 0.10, 0.08, 0.05, 0.03]
REWARD = 0.0
# Advantage of each token: how much the prediction moved across it (GAE with lambda = 0, gamma = 1).
ADV = [b - a for a, b in zip(VALUES, VALUES[1:] + [REWARD])]


def fmt(a):
    if abs(a) < 0.04:  # tiny moves read as noise, show them as 0
        return "0"
    return f"{a:+.2f}".replace("0.", ".").replace("-", "−")


class PPO(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = T("PPO: a critic gives every token its own credit", 32, weight=BOLD).to_edge(UP, buff=0.35)
        prompt = VGroup(T("prompt", 20, DIM), T(QUESTION, 24, font=MONO)).arrange(RIGHT, buff=0.3)
        prompt.next_to(title, DOWN, buff=0.3)

        # 1. the model writes its answer token by token; the final answer is wrong
        boxes = VGroup()
        for tok in TOKENS:
            txt = T(tok, 20, BLUE if tok.startswith("<") else FG, font=MONO)
            box = RoundedRectangle(corner_radius=0.08, width=txt.width + 0.3, height=0.62,
                                   stroke_color=DIM, stroke_width=1.5, fill_color=CARD, fill_opacity=1)
            boxes.add(VGroup(box, txt.move_to(box)))
        boxes.arrange(RIGHT, buff=0.08).move_to([-0.6, 1.55, 0])
        reward = VGroup(T("reward", 18, DIM), T("0", 30, BAD, weight=BOLD)).arrange(DOWN, buff=0.05)
        reward.next_to(boxes, RIGHT, buff=0.35)

        self.play(FadeIn(title), FadeIn(prompt), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * RIGHT) for b in boxes], lag_ratio=0.35), run_time=1.8)
        self.play(FadeIn(reward, scale=1.3), run_time=0.4)
        self.wait(0.3)

        # 2. the critic's prediction at each point in the answer
        y0, y1 = -1.7, 0.3  # plot coordinates for 0 and 1
        xs = [b.get_left()[0] for b in boxes] + [boxes[-1].get_right()[0]]
        ypos = lambda v: y0 + v * (y1 - y0)
        axis = Line([xs[0] - 0.1, y0, 0], [xs[-1] + 0.1, y0, 0], stroke_color=DIM, stroke_width=1.5)
        top = DashedLine([xs[0] - 0.1, y1, 0], [xs[-1] + 0.1, y1, 0], stroke_color=DIM, stroke_width=1,
                         dash_length=0.08)
        ylabels = VGroup(T("0", 16, DIM).next_to(axis, LEFT, buff=0.15), T("1", 16, DIM).next_to(top, LEFT, buff=0.15))
        grid = VGroup(*[DashedLine([x, y0, 0], [x, boxes.get_bottom()[1] - 0.05, 0], stroke_color=DIM,
                                   stroke_width=0.8, stroke_opacity=0.4, dash_length=0.06) for x in xs])
        label = T("critic: chance this ends up correct", 19, ACCENT).move_to([xs[0], y1 + 0.28, 0], aligned_edge=LEFT)
        pts = [[x, ypos(v), 0] for x, v in zip(xs, VALUES + [REWARD])]
        curve = VMobject(stroke_color=ACCENT, stroke_width=4).set_points_as_corners(pts)
        dots = VGroup(*[Dot(p, radius=0.06, color=ACCENT) for p in pts[:-1]], Dot(pts[-1], radius=0.08, color=BAD))

        self.play(Create(axis), Create(top), FadeIn(ylabels), FadeIn(grid), FadeIn(label), run_time=0.7)
        self.play(Create(curve), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.12), run_time=2.0)

        # the drop happens across "272": that's where the answer went wrong
        culprit = SurroundingRectangle(boxes[2], color=BAD, buff=0.06, corner_radius=0.08)
        drop = T("47*6 is 282, not 272", 17, BAD).next_to(Line(pts[2], pts[3]).get_center(), RIGHT, buff=0.3)
        self.play(Create(culprit), FadeIn(drop), run_time=0.7)
        self.wait(0.8)

        # 3. each token's advantage is how much the prediction moved across it
        formula = T("advantage of a token ≈ critic after it − critic before it", 20, ACCENT, font=MONO)
        formula.to_edge(DOWN, buff=0.3)
        advs = VGroup(*[T(fmt(a), 20, BAD if a < -0.05 else DIM, weight=BOLD if a < -0.05 else NORMAL)
                        .move_to([b.get_center()[0], y0 - 0.35, 0]) for a, b in zip(ADV, boxes)])
        adv_label = T("advantage", 17, DIM).next_to(advs, LEFT, buff=0.35).align_to(ylabels, RIGHT)
        self.play(FadeOut(drop), FadeIn(formula), FadeIn(adv_label), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(a, shift=0.15 * DOWN) for a in advs], lag_ratio=0.12),
                  *[b[0].animate.set_fill(BAD, opacity=min(0.9, 0.12 + 1.6 * -a)) for a, b in zip(ADV, boxes) if a < -0.05],
                  run_time=1.4)
        self.wait(1.0)

        takeaway = T("REINFORCE and GRPO blame every token equally.  PPO's critic finds the one that mattered.", 19)
        takeaway.move_to(formula)
        self.play(FadeOut(formula, shift=0.2 * DOWN), FadeIn(takeaway, shift=0.2 * UP), run_time=0.6)
        self.wait(2.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)
