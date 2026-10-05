"""Plots and HTML views used by the notebooks. Nothing here is RL; it's all for looking."""

import difflib
import html
import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Code, display

PALETTE = {"blue": "#2f6db5", "orange": "#e07b2a", "green": "#2e9a5b", "red": "#c94040",
           "purple": "#7a4fb5", "gray": "#7a7a7a"}
ALGO_COLORS = {"reinforce": PALETTE["blue"], "grpo": PALETTE["green"], "ppo": PALETTE["orange"],
               "dpo": PALETTE["purple"], "base": PALETTE["gray"]}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


# ----------------------------------------------------------------------------- code

def show_code(obj, title=None):
    """Show the source of a function/class/module with syntax highlighting."""
    if title:
        display(HTML(f"<b>{html.escape(title)}</b>"))
    display(Code(inspect.getsource(obj), language="python"))


def show_diff(a, b, label_a=None, label_b=None, context=3):
    """Side-by-side diff of two functions/modules (or source strings): what changed from a to b."""
    src_a = a if isinstance(a, str) else inspect.getsource(a)
    src_b = b if isinstance(b, str) else inspect.getsource(b)
    label_a = label_a or getattr(a, "__name__", "before")
    label_b = label_b or getattr(b, "__name__", "after")
    la, lb = src_a.splitlines(), src_b.splitlines()
    rows, added, removed = [], 0, 0
    css = {"equal": ("", ""), "delete": ("#fbe3e3", ""), "insert": ("", "#e2f5e6"), "replace": ("#fbe3e3", "#e2f5e6")}
    for group in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_grouped_opcodes(context):
        rows.append('<tr><td colspan="4" style="background:#eef1f5;color:#888;text-align:center;padding:0">⋯</td></tr>')
        for tag, i1, i2, j1, j2 in group:
            left, right = la[i1:i2], lb[j1:j2]
            if tag in ("delete", "replace"):
                removed += len(left)
            if tag in ("insert", "replace"):
                added += len(right)
            for k in range(max(len(left), len(right))):
                l_txt = html.escape(left[k]) if k < len(left) else ""
                r_txt = html.escape(right[k]) if k < len(right) else ""
                l_no = str(i1 + k + 1) if k < len(left) else ""
                r_no = str(j1 + k + 1) if k < len(right) else ""
                bg_l, bg_r = css[tag]
                bg_l = bg_l if k < len(left) else ("#f4f4f4" if tag != "equal" else "")
                bg_r = bg_r if k < len(right) else ("#f4f4f4" if tag != "equal" else "")
                rows.append(
                    f'<tr><td style="color:#999;text-align:right;padding:0 6px">{l_no}</td>'
                    f'<td style="background:{bg_l};white-space:pre-wrap;word-break:break-all;padding:0 8px">{l_txt}</td>'
                    f'<td style="color:#999;text-align:right;padding:0 6px;border-left:1px solid #ddd">{r_no}</td>'
                    f'<td style="background:{bg_r};white-space:pre-wrap;word-break:break-all;padding:0 8px">{r_txt}</td></tr>'
                )
    header = (f'<div style="font-family:sans-serif;margin:4px 0"><b>{html.escape(label_a)}</b> → '
              f'<b>{html.escape(label_b)}</b>: <span style="color:#2e9a5b">+{added}</span> / '
              f'<span style="color:#c94040">−{removed}</span> lines</div>')
    table = ('<div style="overflow-x:auto;background:#fff;color:#222;border:1px solid #ddd;border-radius:4px">'
             '<table style="border-collapse:collapse;table-layout:fixed;font-family:Menlo,Consolas,monospace;font-size:11.5px;width:100%">'
             '<colgroup><col style="width:3em"><col style="width:calc(50% - 3em)"><col style="width:3em"><col style="width:calc(50% - 3em)"></colgroup>'
             f'<tr style="background:#f6f8fa"><th></th><th style="text-align:left;padding:2px 8px">{html.escape(label_a)}</th>'
             f'<th></th><th style="text-align:left;padding:2px 8px">{html.escape(label_b)}</th></tr>'
             + "".join(rows) + "</table></div>")
    display(HTML(header + table))


# ----------------------------------------------------------------------------- tokens

def _color(value, vmax, pos=(46, 154, 91), neg=(201, 64, 64)):
    if vmax == 0 or value is None or math.isnan(value):
        return "transparent"
    a = min(abs(value) / vmax, 1.0) * 0.85
    r, g, b = pos if value >= 0 else neg
    return f"rgba({r},{g},{b},{a:.2f})"


def show_tokens(tokens, values, title="", vmax=None, fmt="{:+.2f}", caption=""):
    """Render tokens as text with each one shaded by its value: green positive, red negative.
    Hover a token to see its exact value."""
    values = [float(v) for v in values]
    vmax = vmax or max((abs(v) for v in values), default=1.0) or 1.0
    spans = []
    for t, v in zip(tokens, values):
        shown = html.escape(t).replace("\n", "↵<br>")
        spans.append(f'<span title="{fmt.format(v)}" style="background:{_color(v, vmax)};'
                     f'border-radius:3px;padding:1px 0;margin:0 0.5px">{shown}</span>')
    head = f"<div style='font-family:sans-serif;margin-bottom:4px'><b>{html.escape(title)}</b></div>" if title else ""
    cap = f"<div style='font-family:sans-serif;color:#666;font-size:12px;margin-top:4px'>{caption}</div>" if caption else ""
    display(HTML(f"{head}<div style='font-family:Menlo,Consolas,monospace;font-size:13px;line-height:1.9;"
                 f"background:#fff;color:#222;padding:8px;border:1px solid #ddd;border-radius:4px'>"
                 f"{''.join(spans)}</div>{cap}"))


def completion_tokens(tok, rollout, i):
    """(token strings, positions) of row i's completion."""
    pos = rollout.completion_mask[i].nonzero().squeeze(-1).tolist()
    ids = rollout.input_ids[i, pos].tolist()
    return [tok.decode([t]) for t in ids], pos


def show_completions(rows, title=""):
    """rows: list of dicts with keys text, reward and optional advantage / note."""
    cells = []
    for r in rows:
        color = "#2e9a5b" if r.get("reward", 0) > 0 else "#c94040"
        extra = f" · advantage <b>{r['advantage']:+.2f}</b>" if "advantage" in r else ""
        note = f" · {html.escape(r['note'])}" if r.get("note") else ""
        cells.append(
            f"<div style='border-left:4px solid {color};padding:4px 8px;margin:4px 0;background:#fafafa;color:#222'>"
            f"<div style='font-family:sans-serif;font-size:12px;color:#555'>reward <b>{r.get('reward', 0):.0f}</b>{extra}{note}</div>"
            f"<div style='font-family:Menlo,Consolas,monospace;font-size:12px;white-space:pre-wrap'>{html.escape(r['text'])}</div></div>"
        )
    head = f"<div style='font-family:sans-serif'><b>{html.escape(title)}</b></div>" if title else ""
    display(HTML(head + "".join(cells)))


# ----------------------------------------------------------------------------- training curves

def smooth(xs, k=5):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < k:
        return xs
    pad = np.concatenate([np.full(k - 1, xs[0]), xs])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def plot_history(history, keys, titles=None, color=PALETTE["blue"], ncols=4, suptitle=None):
    """Small multiples of logged quantities, raw (faint) and smoothed (solid)."""
    n = len(keys)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows), squeeze=False)
    for ax, key, title in zip(axes.flat, keys, titles or keys):
        ys = [h[key] for h in history if key in h]
        ax.plot(ys, color=color, alpha=0.3, lw=1)
        ax.plot(smooth(ys), color=color, lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


class LivePlot:
    """A figure that redraws in place while a training loop runs."""

    def __init__(self, keys, titles=None, every=1, color=PALETTE["blue"]):
        self.keys, self.titles, self.every, self.color = keys, titles, every, color
        self.handle = display(HTML("<i>training…</i>"), display_id=True)

    def update(self, history, final=False):
        if not final and len(history) % self.every:
            return
        fig = plot_history(history, self.keys, self.titles, color=self.color)
        self.handle.update(fig)
        plt.close(fig)


def compare_runs(runs, keys, titles=None, x="step"):
    """Overlay the same metric from several runs (dict name -> history)."""
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), squeeze=False)
    for ax, key, title in zip(axes.flat, keys, titles or keys):
        for name, history in runs.items():
            ys = [h[key] for h in history if key in h]
            if not ys:
                continue
            xs = np.cumsum([h["generated_tokens"] for h in history]) / 1e3 if x == "tokens" else np.arange(len(ys))
            c = ALGO_COLORS.get(name, None)
            ax.plot(xs[:len(ys)], ys, color=c, alpha=0.25, lw=1)
            ax.plot(xs[:len(ys)], smooth(ys), color=c, lw=2, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("thousand generated tokens" if x == "tokens" else "step")
    axes.flat[0].legend()
    fig.tight_layout()
    return fig


def figure(path, width=760):
    """Show one of the SVG diagrams in notebooks/figures/."""
    from pathlib import Path
    svg = (Path(__file__).resolve().parent.parent / "notebooks" / "figures" / path).read_text()
    display(HTML(f"<div style='max-width:{width}px'>{svg}</div>"))
