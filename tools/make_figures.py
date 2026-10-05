"""Generate the per-algorithm diagrams in notebooks/figures/ (run: python tools/make_figures.py)."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
STYLE = {
    "gray": ("#f3f3f3", "#777", "#333"),
    "blue": ("#e8f0fb", "#2f6db5", "#1d3f6e"),
    "green": ("#e6f4ec", "#2e9a5b", "#1b5e37"),
    "orange": ("#fdf0e4", "#e07b2a", "#7a3f0f"),
    "purple": ("#f1eafa", "#7a4fb5", "#47286f"),
    "red": ("#fbe7e7", "#c94040", "#7a1f1f"),
}


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def box(self, x, y, w, h, lines, color="gray", bold_first=True, dashed=False, size=13):
        fill, stroke, text = STYLE[color]
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        n = len(lines)
        for i, line in enumerate(lines):
            ty = y + h / 2 + (i - (n - 1) / 2) * (size + 5) + size * 0.35
            weight = ' font-weight="bold"' if bold_first and i == 0 else ""
            self.parts.append(f'<text x="{x + w / 2}" y="{ty:.1f}" text-anchor="middle" fill="{text}" font-size="{size}"{weight}>{escape(line)}</text>')

    def arrow(self, x1, y1, x2, y2, label=None, dy=-6, color="#555"):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#a)"/>')
        if label:
            self.parts.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" text-anchor="middle" fill="#777" font-size="11">{escape(label)}</text>')

    def text(self, x, y, s, color="#333", size=13, anchor="start", bold=False):
        weight = ' font-weight="bold"' if bold else ""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}"{weight}>{escape(s)}</text>')

    def chip(self, x, y, label, color, note):
        fill, stroke, text = STYLE[color]
        w = 16 + 9 * len(label)
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.text(x + w / 2, y + 17, label, text, 13, "middle", True)
        self.text(x + w + 8, y + 17, note, "#555", 12)
        return w + 8 + 6.5 * len(note) + 18

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                'font-family="Helvetica, Arial, sans-serif">'
                '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        (OUT / name).write_text(head + "\n".join(self.parts) + "</svg>\n")


def pipeline(svg, y, steps):
    """Four boxes left to right with arrows between them."""
    x, w, gap = 20, 165, 25
    for i, (lines, color) in enumerate(steps):
        svg.box(x + i * (w + gap), y, w, 76, lines, color)
        if i:
            svg.arrow(x + i * (w + gap) - gap, y + 38, x + i * (w + gap) - 2, y + 38)


def models(svg, y, items):
    svg.text(20, y - 8, "Models in GPU memory", "#555", 12, bold=True)
    x = 20
    for label, color, note in items:
        x += svg.chip(x, y, label, color, note)


def reinforce():
    s = Svg(760, 220)
    s.text(20, 26, "REINFORCE: one gradient step per batch, then sample again", bold=True)
    pipeline(s, 45, [
        (["1 · Sample", "16 prompts × 4", "completions"], "gray"),
        (["2 · Score", "reward 0 or 1", "per completion"], "gray"),
        (["3 · Advantage", "A = r − running", "average of rewards"], "orange"),
        (["4 · Update (×1)", "−A · log π(token)", "+ β · KL(π ‖ π_ref)"], "green"),
    ])
    s.arrow(712, 125, 712, 150, None)
    s.text(700, 165, "repeat", "#777", 11, "end")
    models(s, 182, [("π", "blue", "policy (trained)"), ("π_ref", "gray", "frozen copy of the start")])
    s.save("reinforce.svg")


def grpo():
    s = Svg(760, 355)
    s.text(20, 26, "GRPO: the baseline is the group; each batch is used for several clipped updates", bold=True)
    pipeline(s, 45, [
        (["1 · Sample", "a GROUP of 4", "per prompt"], "gray"),
        (["2 · Score", "reward 0 or 1", "per completion"], "gray"),
        (["3 · Advantage", "A = (r − group mean)", "/ group std"], "orange"),
        (["4 · Update (×2)", "clip(π/π_old) · A", "+ β · KL(π ‖ π_ref)"], "green"),
    ])
    # one group example
    s.text(20, 160, "One prompt's group:", "#555", 12, bold=True)
    ex = [("✓ …<answer>109</answer>", 1, "+0.87"), ("✗ …<answer>99</answer>", 0, "−0.87"),
          ("✗ …(no answer tag)", 0, "−0.87"), ("✓ …<answer>109</answer>", 1, "+0.87")]
    for i, (t, r, a) in enumerate(ex):
        y = 172 + i * 28
        s.box(20, y, 260, 24, [t], "green" if r else "red", bold_first=False, size=12)
        s.text(295, y + 17, f"reward {r}", "#555", 12)
        s.text(370, y + 17, f"advantage {a}", "#1b5e37" if r else "#7a1f1f", 12, bold=True)
    s.text(480, 190, "mean 0.5, std 0.58:", "#777", 12)
    s.text(480, 208, "right answers pushed up,", "#777", 12)
    s.text(480, 226, "wrong ones pushed down.", "#777", 12)
    s.text(480, 250, "All 4 right (or all wrong)?", "#777", 12)
    s.text(480, 268, "Every A = 0: nothing learned.", "#777", 12)
    models(s, 325, [("π", "blue", "policy (trained)"), ("π_old", "blue", "π when it sampled (just saved log-probs)"),
                    ("π_ref", "gray", "frozen start")])
    s.save("grpo.svg")


def ppo():
    s = Svg(760, 330)
    s.text(20, 26, "PPO: a learned critic gives every token its own advantage", bold=True)
    pipeline(s, 45, [
        (["1 · Sample", "1 completion", "per prompt"], "gray"),
        (["2 · Score + critic", "reward at the end,", "V(t) at every token"], "purple"),
        (["3 · Advantage (GAE)", "A_t from changes", "in V along the text"], "orange"),
        (["4 · Update (×2)", "π: clip(π/π_old) · A_t", "V: regress to returns"], "green"),
    ])
    s.text(20, 160, "The critic reads the reply token by token and keeps guessing P(correct):", "#555", 12, bold=True)
    toks = ["23*4", "=82", "+17", "=99", "<answer>99"]
    vals = [0.55, 0.30, 0.28, 0.25, 0.05]
    for i, (t, v) in enumerate(zip(toks, vals)):
        x = 30 + i * 110
        s.box(x, 175, 96, 28, [t], "blue", bold_first=False, size=12)
        h = v * 70
        s.parts.append(f'<rect x="{x + 33}" y="{280 - h}" width="30" height="{h}" fill="#7a4fb5" opacity="0.75"/>')
        s.text(x + 48, 295, f"V={v:.2f}", "#47286f", 11, "middle")
    s.text(590, 200, "The big drop at \"=82\"", "#777", 12)
    s.text(590, 218, "(the arithmetic slip)", "#777", 12)
    s.text(590, 236, "gets the most blame:", "#777", 12)
    s.text(590, 254, "credit per token,", "#777", 12)
    s.text(590, 272, "not per completion.", "#777", 12)
    models(s, 335, [("π", "blue", "policy"), ("π_old", "blue", "sampler"), ("π_ref", "gray", "frozen start"),
                    ("V", "purple", "critic: a 2nd trained LLM")])
    s.h = 368
    s.save("ppo.svg")


def dpo():
    s = Svg(760, 250)
    s.text(20, 26, "DPO: offline. Sample once, then train on fixed preference pairs", bold=True)
    s.box(20, 45, 165, 76, ["Once, up front", "π_ref samples 4", "replies per prompt"], "gray", dashed=True)
    s.box(210, 45, 165, 76, ["Make pairs", "chosen = a ✓ reply", "rejected = a ✗ reply"], "gray", dashed=True)
    s.arrow(185, 83, 208, 83)
    s.box(400, 45, 340, 76, ["Training loop (no sampling, no reward)",
                             "margin = β[(log π − log π_ref)(✓) − (same)(✗)]",
                             "loss = −log σ(margin)"], "green")
    s.arrow(375, 83, 398, 83)
    s.text(400, 150, "Pushes log π(✓) up and log π(✗) down, relative to π_ref.", "#555", 12)
    s.text(400, 168, "Can only learn from replies already in the dataset:", "#555", 12)
    s.text(400, 186, "prompts with no ✓ (or no ✗) reply give no pair at all.", "#555", 12)
    models(s, 222, [("π", "blue", "policy (trained)"), ("π_ref", "gray", "frozen start (log-probs precomputed)")])
    s.save("dpo.svg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (reinforce, grpo, ppo, dpo):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
