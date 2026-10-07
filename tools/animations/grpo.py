"""GRPO in one picture: score each answer against the other answers to the same prompt.

Render (needs `pip install manim`, no LaTeX):
    manim -r 960,540 --fps 15 tools/animations/grpo.py GRPO
then convert to a GIF with tools/animations/to_gif.sh.
"""

from manim import *

BG = "#161b22"
FG = "#e6edf3"
DIM = "#8b949e"
GOOD = "#3fb950"
BAD = "#f85149"
ACCENT = "#e3a33b"  # the group / baseline colour (orange, as in the notebook diagrams)
BLUE = "#58a6ff"

SANS = "Inter"
MONO = "DejaVu Sans Mono"

QUESTION = "What is 47 * 6 + 35?"
# (reasoning, answer, reward). Rewards 1,0,0,0 give mean 0.25, std 0.5, advantages +1.5 / -0.5.
COMPLETIONS = [
    ("47*6 = 282, 282+35 = 317", "317", 1),
    ("47*6 = 272, 272+35 = 307", "307", 0),
    ("6+35 = 41, 47*41 = 1927", "1927", 0),
    ("47*6 = 282, 282+35 = 307", "307", 0),
]
ADV = [1.5, -0.5, -0.5, -0.5]


def T(s, size=24, color=FG, font=SANS, weight=NORMAL):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


class GRPO(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = T("GRPO: grade each answer against its own group", 32, weight=BOLD).to_edge(UP, buff=0.35)
        prompt = VGroup(T("prompt", 20, DIM), T(QUESTION, 24, font=MONO)).arrange(RIGHT, buff=0.3)
        prompt.next_to(title, DOWN, buff=0.35)

        ys = [1.0, 0.05, -0.9, -1.85]
        x_ans, x_rew, x_adv, x_prob = -3.55, 1.0, 2.8, 5.0

        heads = VGroup(
            T("4 sampled answers", 20, DIM).move_to([x_ans, 1.8, 0]),
            T("reward", 20, DIM).move_to([x_rew, 1.8, 0]),
            T("advantage", 20, DIM).move_to([x_adv, 1.8, 0]),
            T("probability", 20, DIM).move_to([x_prob, 1.8, 0]),
        )

        # 1. sample a group of answers
        self.play(FadeIn(title), FadeIn(prompt), run_time=0.8)
        rows = VGroup()
        for (reason, ans, _), y in zip(COMPLETIONS, ys):
            box = RoundedRectangle(corner_radius=0.12, width=6.6, height=0.72,
                                   stroke_color=DIM, stroke_width=1.5, fill_color="#21262d", fill_opacity=1)
            txt = T(reason, 17, font=MONO)
            tag = T(f"<answer>{ans}</answer>", 17, BLUE, font=MONO)
            line = VGroup(txt, tag).arrange(RIGHT, buff=0.2)
            line.scale_to_fit_width(min(line.width, box.width - 0.4)).move_to(box)
            rows.add(VGroup(box, txt, tag).move_to([x_ans, y, 0]))
        self.play(FadeIn(heads[0]), LaggedStart(*[FadeIn(r, shift=0.3 * RIGHT) for r in rows], lag_ratio=0.25),
                  run_time=1.6)
        self.wait(0.4)

        # 2. score them: 1 if the number is right, else 0
        rewards = VGroup()
        for (_, _, r), row, y in zip(COMPLETIONS, rows, ys):
            rewards.add(T(str(r), 30, GOOD if r else BAD, weight=BOLD).move_to([x_rew, y, 0]))
        self.play(FadeIn(heads[1]), run_time=0.3)
        for (_, _, r), row, rew in zip(COMPLETIONS, rows, rewards):
            self.play(row[0].animate.set_stroke(GOOD if r else BAD, 2.5), FadeIn(rew, scale=1.4), run_time=0.35)
        self.wait(0.4)

        # 3. the group's own mean and std are the baseline
        group = SurroundingRectangle(rewards, color=ACCENT, buff=0.22, corner_radius=0.1)
        stats = VGroup(T("group mean 0.25", 20, ACCENT), T("group std 0.5", 20, ACCENT)).arrange(DOWN, buff=0.1)
        stats.next_to(group, DOWN, buff=0.18)
        self.play(Create(group), FadeIn(stats, shift=0.2 * UP), run_time=0.9)
        self.wait(0.5)

        # 4. advantage = (reward - mean) / std
        formula = T("advantage = (reward − mean) / std", 22, ACCENT, font=MONO).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(heads[2]), FadeIn(formula), run_time=0.5)
        advs = VGroup()
        for a, y in zip(ADV, ys):
            advs.add(T(f"{a:+.1f}", 30, GOOD if a > 0 else BAD, weight=BOLD).move_to([x_adv, y, 0]))
        self.play(LaggedStart(*[TransformFromCopy(r, a) for r, a in zip(rewards, advs)], lag_ratio=0.2),
                  run_time=1.3)
        self.wait(0.7)

        # 5. push probability up for above-average answers, down for below-average ones
        w0, h = 1.0, 0.36
        bars, arrows = VGroup(), VGroup()
        for y in ys:
            bars.add(Rectangle(width=w0, height=h, stroke_width=0, fill_color=BLUE, fill_opacity=0.85)
                     .move_to([x_prob - 0.6, y, 0], aligned_edge=LEFT))
        self.play(FadeIn(heads[3]), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1),
                  run_time=0.8)
        new_w = [2.0, 0.6, 0.6, 0.6]
        targets = VGroup(*[
            Rectangle(width=w, height=h, stroke_width=0, fill_color=GOOD if a > 0 else BAD, fill_opacity=0.9)
            .move_to([x_prob - 0.6, y, 0], aligned_edge=LEFT)
            for w, a, y in zip(new_w, ADV, ys)])
        for a, t in zip(ADV, targets):
            arrows.add(T("▲" if a > 0 else "▼", 22, GOOD if a > 0 else BAD).next_to(t, RIGHT, buff=0.12))
        self.play(*[Transform(b, t) for b, t in zip(bars, targets)], FadeIn(arrows), run_time=1.4)
        self.wait(0.5)

        takeaway = T("Better than its group → more likely.  Worse → less likely.  No critic needed.", 22)
        takeaway.move_to(formula)
        self.play(FadeOut(formula, shift=0.2 * DOWN), FadeIn(takeaway, shift=0.2 * UP), run_time=0.6)
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)
