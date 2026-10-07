"""Notebook 4: GRPO."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 4 · GRPO: group baselines and clipped updates

**Run-only. Needs the GPU; training takes roughly 20–40 minutes.**

GRPO (from DeepSeekMath, and the algorithm behind DeepSeek-R1) is REINFORCE with two changes:

1. **The baseline is the group.** Sample several completions for the same prompt; each one's advantage is how it did *relative to its siblings*. No running average, no critic.
2. **Each batch is used for several updates.** Generation is the slow part, so squeeze more out of it. To do that safely, weight each token by the ratio π_new / π_old and **clip** that ratio. That's the core idea of PPO, borrowed without PPO's critic.
"""),
    code(r"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from rlcourse import llm, task, viz, reinforce, grpo

viz.figure("grpo.svg")
"""),
    md(r"""
## What changed from REINFORCE

The full diff between `reinforce.py` and `grpo.py`. Green lines are new in GRPO, red lines are gone.
"""),
    code(r"""
import inspect
viz.show_diff(inspect.getsource(reinforce), inspect.getsource(grpo), "reinforce.py", "grpo.py")
"""),
    md(r"""
## Change 1: the group-relative advantage

Here it is on a real batch from the starting model: 8 prompts × 4 samples. Each column is one prompt's group; dots are the 4 completions, colored by their advantage.
"""),
    code(r"""
tok = llm.load_tokenizer()
model = llm.load_model(trainable=True)
ref_model = llm.load_model(trainable=False)
cfg = llm.maybe_smoke(grpo.Config())

torch.manual_seed(0)
demo = llm.generate(model, tok, task.TrainStream(cfg.difficulty, seed=7).next(8), cfg.samples_per_prompt, cfg.max_new_tokens)
adv = grpo.advantages(demo.rewards, demo.group, {}, cfg)

fig, ax = plt.subplots(figsize=(9, 3))
k = cfg.samples_per_prompt
for g in range(8):
    rs, a = demo.rewards[g * k:(g + 1) * k], adv[g * k:(g + 1) * k]
    jitter = np.linspace(-0.2, 0.2, k)
    sc = ax.scatter(g + jitter, rs + 0.03 * jitter, c=a, cmap="RdYlGn", vmin=-1.5, vmax=1.5, s=80, edgecolors="black")
    if rs.std() == 0:
        ax.text(g, 0.5, "all same:\nA = 0", ha="center", fontsize=8, color="gray")
ax.set_xticks(range(8)), ax.set_xticklabels([f"prompt {g + 1}" for g in range(8)])
ax.set_yticks([0, 1]), ax.set_yticklabels(["wrong (0)", "right (1)"])
plt.colorbar(sc, label="advantage"), ax.set_title("Rewards and group-relative advantages"), plt.show()
"""),
    code(r"""
g = int(next((i for i in range(8) if 0 < demo.rewards[i * k:(i + 1) * k].sum() < k), 0))
rows = range(g * k, (g + 1) * k)
viz.show_completions([{"text": demo.texts[i], "reward": demo.rewards[i].item(), "advantage": adv[i].item()} for i in rows],
                     title=f"One group: {demo.problems[g * k].question}  (answer {demo.problems[g * k].answer})")
"""),
    md(r"""
## Change 2: the ratio and the clip

After the first gradient step on a batch, the model has changed, but the samples came from the old one (π_old). The ratio $\rho = \pi_\text{new}(\text{token}) / \pi_\text{old}(\text{token})$ says how much more (or less) likely each token has become since sampling. The loss uses $\rho \cdot A$, clipped:

$$ \text{objective} = \min\big(\rho A,\ \text{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon)\, A\big) $$

The picture says it best. Once a token's probability has moved by more than ε (20%) **in the direction its advantage wants**, the objective goes flat: no more gradient to push it further. In the other direction there's no cap, so mistakes can always be undone.
"""),
    code(r"""
rho = np.linspace(0.4, 1.6, 300)
eps = cfg.clip_eps
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=False)
for ax, A, title in [(axes[0], 1.0, "advantage > 0 (a good completion)"), (axes[1], -1.0, "advantage < 0 (a bad completion)")]:
    unclipped = rho * A
    clipped = np.clip(rho, 1 - eps, 1 + eps) * A
    ax.plot(rho, unclipped, "--", color="gray", label="ρ·A (no clipping)")
    ax.plot(rho, np.minimum(unclipped, clipped), color=viz.PALETTE["green"], lw=3, label="GRPO / PPO objective")
    flat = (rho > 1 + eps) if A > 0 else (rho < 1 - eps)
    ax.fill_between(rho, *ax.get_ylim(), where=flat, color=viz.PALETTE["orange"], alpha=0.15, label="flat: zero gradient")
    ax.axvline(1, color="black", lw=0.7)
    ax.set_xlabel("ρ = π_new / π_old"), ax.set_title(title), ax.legend(fontsize=8)
fig.tight_layout(), plt.show()
"""),
    md(r"""
On the **first** update of each batch, the model hasn't changed yet, so ρ = 1 everywhere, the clip does nothing, and the step is exactly REINFORCE with a group baseline. The clip only matters on the second and later updates. We use `updates_per_batch = 2`.

The new loss, side by side with REINFORCE's:
"""),
    code(r"""
viz.show_diff(reinforce.policy_loss, grpo.policy_loss, "REINFORCE policy_loss", "GRPO policy_loss")
"""),
    md(r"""
Note the three different "versions" of the model now in play:

| | what it is | role |
|---|---|---|
| **π** | the model being trained | gets gradients |
| **π_old** | π at the moment it sampled this batch (we just store its log-probs) | denominator of the ratio: "how far have we moved since sampling?" |
| **π_ref** | frozen copy of the starting model | KL leash: "how far have we moved since the start?" |

## Run it

Same budget as REINFORCE: 16 prompts × 4 samples per step, same learning rate and KL coefficient. Two new things to watch: **clip fraction** (share of tokens where the clip was active on the second update) and **zero-variance groups** (share of prompts whose 4 samples all got the same reward, and so taught nothing).
"""),
    code(r"""
import time
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
live = viz.LivePlot(["reward", "kl", "clip_frac", "zero_variance_groups", "completion_tokens"],
                    ["reward (train accuracy)", "KL to reference", "clip fraction", "zero-variance groups", "completion length"],
                    every=2, color=viz.ALGO_COLORS["grpo"])
t0 = time.time()
history, samples = grpo.train(model, ref_model, tok, cfg, callback=live.update)
live.update(history, final=True)
minutes = (time.time() - t0) / 60
print(f"{minutes:.1f} minutes, peak GPU memory {llm.peak_memory_gb():.1f} GB")
"""),
    md(r"""
**Notice** the zero-variance curve. As the model gets better, more groups are all-correct and contribute nothing. That's GRPO's built-in curriculum problem: it learns most from prompts it gets right *sometimes*. (Variants like DAPO drop such groups and sample new prompts to fill the batch.)

## REINFORCE vs GRPO

If you ran notebook 3, here are both runs on the same axes. The x-axis is generated tokens, the real cost.
"""),
    code(r"""
runs = {"grpo": history}
prev = llm.load_run("reinforce")
if prev:
    runs = {"reinforce": prev["history"], "grpo": history}
viz.compare_runs(runs, ["reward", "kl"], ["reward (train accuracy)", "KL to reference"], x="tokens")
plt.show()
"""),
    code(r"""
for s in (samples[0], samples[-1]):
    viz.show_completions(s["completions"], title=f"step {s['step']}: {s['question']}")
"""),
    code(r"""
results = llm.evaluate(model, tok)
results.update(minutes=minutes, peak_memory_gb=llm.peak_memory_gb())
llm.save_run("grpo", cfg, history, results, samples)

names = [n for n in ["base", "reinforce", "grpo"] if llm.load_run(n)]
accs = [llm.load_run(n)["results"]["accuracy"] for n in names]
plt.figure(figsize=(5, 2.8))
plt.bar(names, accs, color=[viz.ALGO_COLORS[n] for n in names])
for i, a in enumerate(accs):
    plt.text(i, a + 0.01, f"{a:.0%}", ha="center")
plt.ylabel("held-out accuracy (greedy)"), plt.ylim(0, 1), plt.show()
"""),
    md(r"""
## What to take away

- GRPO = REINFORCE + a **group baseline** (compare siblings for the same prompt) + **clipped reuse** of each batch.
- The group baseline is cheap and adapts per prompt: an easy prompt's samples are compared with each other, not with a global average.
- All-correct or all-wrong groups teach nothing. Task difficulty directly controls how much signal you get.
- The ratio and clip only matter after the first update on a batch. They're what make reuse safe.
- Still one advantage per completion, shared by all its tokens. PPO's critic changes that next.
"""),
]

CELLS.append(md(r"""
## Further reading

- Z. Shao et al. (2024), [DeepSeekMath](https://arxiv.org/abs/2402.03300): introduces GRPO.
- DeepSeek-AI (2025), [DeepSeek-R1](https://arxiv.org/abs/2501.12948): GRPO with verifiable rewards at scale.
- Z. Liu et al. (2025), [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (Dr. GRPO): the length and std-normalization biases.
- Q. Yu et al. (2025), [DAPO](https://arxiv.org/abs/2503.14476): token-level loss averaging, dynamic sampling of zero-variance groups.
"""))
