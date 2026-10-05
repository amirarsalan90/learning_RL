"""Notebook 7: all algorithms side by side."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 7 · Side by side

**Run-only, no GPU needed.** This reads the results that notebooks 2–6 saved in `runs/` and puts them on the same axes. Run it after the others; any you skipped are simply left out.

Keep in mind these are single runs with short budgets and hand-picked settings. They're evidence about *this* experiment, not a ranking of algorithms.
"""),
    code(r"""
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from rlcourse import llm, viz, reinforce, grpo, ppo, dpo

names = ["base", "reinforce", "grpo", "ppo", "dpo"]
runs = {n: llm.load_run(n) for n in names}
runs = {n: r for n, r in runs.items() if r}
print("found runs:", ", ".join(runs))
"""),
    md(r"""
## The summary table
"""),
    code(r"""
rows = ["| | held-out accuracy | well-formed answers | generated tokens (k) | minutes | peak GPU GB |", "|---|---|---|---|---|---|"]
for n, r in runs.items():
    res, hist = r["results"], r["history"]
    tokens = res.get("generated_tokens") or sum(h.get("generated_tokens", 0) for h in hist)
    rows.append(f"| **{n}** | {res['accuracy']:.1%} | {res['format_rate']:.1%} | {tokens / 1e3:,.0f} | "
                f"{res.get('minutes', float('nan')):.1f} | {res.get('peak_memory_gb', float('nan')):.1f} |")
display(Markdown("\n".join(rows)))
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(7, 3))
accs = [runs[n]["results"]["accuracy"] for n in runs]
ax.bar(list(runs), accs, color=[viz.ALGO_COLORS[n] for n in runs])
for i, a in enumerate(accs):
    ax.text(i, a + 0.01, f"{a:.0%}", ha="center")
ax.set_ylim(0, 1), ax.set_ylabel("held-out accuracy (greedy)"), ax.set_title("Same model, same task, same eval")
plt.show()
"""),
    md(r"""
## Learning curves (online methods)

Training reward against **generated tokens**, the real cost, since generation dominates the runtime. PPO uses 64 prompts × 1 sample, the others 16 prompts × 4 samples, so per step they generate a similar amount.
"""),
    code(r"""
online = {n: runs[n]["history"] for n in ["reinforce", "grpo", "ppo"] if n in runs}
if online:
    viz.compare_runs(online, ["reward", "kl", "completion_tokens"],
                     ["reward (train accuracy)", "KL to reference", "completion length"], x="tokens")
    plt.show()
"""),
    md(r"""
## The same question, every model

Greedy answers to the same held-out problems.
"""),
    code(r"""
for i in range(3):
    entries = []
    for n, r in runs.items():
        ex = r["results"]["examples"][i]
        entries.append({"text": ex["completion"], "reward": ex["reward"], "note": n})
    q = next(iter(runs.values()))["results"]["examples"][i]
    viz.show_completions(entries, title=f"{q['question']}  (answer {q['answer']})")
"""),
    md(r"""
## How the four losses differ, at a glance

The policy losses next to each other. REINFORCE → GRPO adds the ratio and clip; GRPO → PPO moves the KL out of the loss and makes the advantage per-token; DPO is a different shape altogether (pairs, no advantage).
"""),
    code(r"""
viz.show_diff(reinforce.policy_loss, grpo.policy_loss, "REINFORCE", "GRPO")
viz.show_diff(grpo.policy_loss, ppo.policy_loss, "GRPO", "PPO")
viz.show_code(dpo.dpo_loss, "DPO")
"""),
    md(r"""
## Cheat sheet

| | REINFORCE | GRPO | PPO | DPO |
|---|---|---|---|---|
| data | online samples | online, a group per prompt | online, 1 per prompt | fixed pairs, offline |
| baseline / advantage | running mean of rewards | group mean (÷ std) | learned critic + GAE | implicit, via π/π_ref |
| credit per | completion | completion | **token** | completion pair |
| updates per batch | 1 | several (clipped ratio) | several (clipped ratio) | n/a (epochs over the data) |
| KL to the start | in the loss | in the loss | in the reward | built into the loss |
| models in memory | π, π_ref | π, π_ref | π, π_ref, **critic** | π, π_ref |
| needs a reward function | yes | yes | yes | no, needs preference pairs |
| main weakness | noisy, wasteful | prompts with all-equal rewards teach nothing | memory, critic must learn first | can't go beyond its data |
"""),
]
