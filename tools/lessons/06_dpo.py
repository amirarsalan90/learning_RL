"""Notebook 6: DPO, the offline contrast."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 6 · DPO: learning from preference pairs, no sampling loop

**Run-only. Needs the GPU; about 10–20 minutes (most of it building the dataset).**

Everything so far was **online**: sample from the current model, score, update, repeat. DPO is **offline**: you start from a fixed dataset of pairs (prompt, a better reply, a worse reply) and train on it like supervised learning. No reward function inside the loop, no critic, no generation during training.

That's why DPO became the default for preference tuning (where the data is human "I prefer A over B" labels): it's simple and stable. The price: it can only learn from replies already in the dataset.
"""),
    code(r"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from rlcourse import llm, task, viz, dpo

viz.figure("dpo.svg")
"""),
    md(r"""
## Where the loss comes from (in one paragraph)

PPO maximizes *reward − β · KL(π ‖ π_ref)*. That objective has a known best policy: $\pi^*(y) \propto \pi_\text{ref}(y)\, e^{r(y)/\beta}$. Flip it around and the reward can be written using the policy itself: $r(y) = \beta \log \frac{\pi(y)}{\pi_\text{ref}(y)} + \text{const}$. That's the **implicit reward**. Plug it into the standard model of preferences (P(chosen beats rejected) = σ(r_chosen − r_rejected)), and the constant cancels. What's left is a loss on log-probs only:
"""),
    code(r"""
viz.show_code(dpo.dpo_loss)
"""),
    code(r"""
m = np.linspace(-4, 4, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
axes[0].plot(m, -np.log(1 / (1 + np.exp(-m))), color=viz.PALETTE["purple"], lw=2.5)
axes[0].set_xlabel("margin = implicit reward(chosen) − implicit reward(rejected)"), axes[0].set_title("DPO loss = −log σ(margin)")
axes[1].plot(m, 1 / (1 + np.exp(m)), color=viz.PALETTE["purple"], lw=2.5)
axes[1].set_xlabel("margin"), axes[1].set_title("gradient weight σ(−margin)")
fig.tight_layout(), plt.show()
"""),
    md(r"""
**Notice** the right plot: pairs the model already ranks correctly (big positive margin) get almost no gradient; mis-ranked pairs get the most. Like an advantage, it focuses learning where the model is wrong.

## Build the dataset

We make our own preference pairs from the task: sample 4 replies per prompt from the *starting* model, and pair a correct one (chosen) with an incorrect one (rejected). Prompts where all 4 were right, or all 4 wrong, give no pair: notebook 1's "no signal" problem again, now as missing data.
"""),
    code(r"""
tok = llm.load_tokenizer()
model = llm.load_model(trainable=True)
ref_model = llm.load_model(trainable=False)
cfg = llm.maybe_smoke(dpo.Config())
print(cfg)

import time
t0 = time.time()
torch.manual_seed(cfg.seed)
pairs, stats = dpo.build_pairs(ref_model, tok, cfg)
build_minutes = (time.time() - t0) / 60
print(stats, f"({build_minutes:.1f} min)")

plt.figure(figsize=(6, 2.2))
parts = [("usable pairs", stats["pairs"], viz.PALETTE["green"]), ("all 4 right", stats["all_right"], viz.PALETTE["gray"]),
         ("all 4 wrong", stats["all_wrong"], viz.PALETTE["red"])]
left = 0
for label, n, c in parts:
    plt.barh(0, n, left=left, color=c, label=f"{label}: {n}")
    left += n
plt.yticks([]), plt.xlabel("prompts"), plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3), plt.title("What happened to each prompt")
plt.show()
"""),
    code(r"""
import json
(llm.RUNS / "dpo").mkdir(parents=True, exist_ok=True)  # notebook 8 reuses these pairs for TRL
(llm.RUNS / "dpo" / "pairs.json").write_text(json.dumps(
    [{"question": p["question"], "chosen": p["chosen"]["text"], "rejected": p["rejected"]["text"]} for p in pairs]))

p = pairs[0]
viz.show_completions([{"text": p["chosen"]["text"], "reward": 1, "note": "chosen"},
                      {"text": p["rejected"]["text"], "reward": 0, "note": "rejected"}], title=p["question"])
"""),
    md(r"""
## The training loop

Compare it with REINFORCE's `train`: there's no `generate` call and no reward. The reference model's log-probs are computed once, up front, because they never change.
"""),
    code(r"""
viz.show_code(dpo.train)
"""),
    md(r"""
## Run it

Watch the **implicit rewards** β·(log π − log π_ref) for chosen and rejected replies. The loss only cares about their *difference* (the margin). A well-known DPO quirk you can often see here: both can go *down*, with the rejected one just going down faster. The model gets "better" at the preference while making even the chosen replies less likely.
"""),
    code(r"""
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
live = viz.LivePlot(["loss", "pref_accuracy", "margin", "chosen_reward", "rejected_reward"],
                    ["DPO loss", "pairs ranked correctly", "margin", "implicit reward: chosen", "implicit reward: rejected"],
                    every=2, color=viz.ALGO_COLORS["dpo"])
t0 = time.time()
history = dpo.train(model, ref_model, tok, pairs, cfg, callback=live.update)
live.update(history, final=True)
minutes = (time.time() - t0) / 60 + build_minutes
print(f"{minutes:.1f} minutes including dataset, peak GPU memory {llm.peak_memory_gb():.1f} GB")
"""),
    code(r"""
steps = range(len(history))
plt.figure(figsize=(7, 3))
plt.plot(steps, viz.smooth([h["chosen_reward"] for h in history]), color=viz.PALETTE["green"], lw=2, label="chosen")
plt.plot(steps, viz.smooth([h["rejected_reward"] for h in history]), color=viz.PALETTE["red"], lw=2, label="rejected")
plt.axhline(0, color="black", lw=0.6)
plt.xlabel("step"), plt.ylabel("β · log(π / π_ref)"), plt.title("Implicit rewards move apart"), plt.legend(), plt.show()
"""),
    code(r"""
results = llm.evaluate(model, tok)
results.update(minutes=minutes, peak_memory_gb=llm.peak_memory_gb(), generated_tokens=stats["generated_tokens"], pairs=stats["pairs"])
llm.save_run("dpo", cfg, history, results, [])

names = [n for n in ["base", "reinforce", "grpo", "ppo", "dpo"] if llm.load_run(n)]
accs = [llm.load_run(n)["results"]["accuracy"] for n in names]
plt.figure(figsize=(6, 2.8))
plt.bar(names, accs, color=[viz.ALGO_COLORS[n] for n in names])
for i, a in enumerate(accs):
    plt.text(i, a + 0.01, f"{a:.0%}", ha="center")
plt.ylabel("held-out accuracy (greedy)"), plt.ylim(0, 1), plt.show()
"""),
    md(r"""
## What to take away

- DPO optimizes the same KL-regularized objective as PPO, but solves it in closed form, so no reward model, critic or sampling loop is needed.
- It learns only from the pairs it's given. Here those came from the starting model, so as the model improves, the data doesn't follow it. Online methods keep generating fresh, on-policy samples, which is why they usually go further on verifiable tasks like this one.
- The margin can grow while both chosen and rejected log-probs fall. Watch the individual implicit rewards, not just the loss.
- Common in practice: DPO for preference/style data (human labels), GRPO-style online RL for tasks with a checkable reward.
"""),
]

CELLS.append(md(r"""
## Further reading

- R. Rafailov et al. (2023), [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): DPO and its derivation.
"""))
