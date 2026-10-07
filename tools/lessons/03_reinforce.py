"""Notebook 3: REINFORCE on the LLM."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 3 · REINFORCE on an LLM

**Run-only. Needs the GPU; training takes roughly 15–30 minutes.**

This is notebook 1's algorithm, unchanged in spirit, applied to Qwen:
"""),
    code(r"""
import torch
import matplotlib.pyplot as plt
from rlcourse import llm, task, viz, reinforce

viz.figure("reinforce.svg")
"""),
    md(r"""
## The code, piece by piece

The whole algorithm is `rlcourse/reinforce.py`, about 100 lines. The settings:
"""),
    code(r"""
viz.show_code(reinforce.Config)
"""),
    md(r"""
**The advantage.** Notebook 1's baseline idea. Here the baseline is a running average of the rewards seen in *previous* batches. It's computed before this batch's rewards are folded in, so it doesn't depend on the samples it's scoring.
"""),
    code(r"""
viz.show_code(reinforce.advantages)
"""),
    md(r"""
**The loss.** Two parts, per completion token:

1. `-adv * logp`: notebook 1's REINFORCE loss. Every token of a completion gets the *same* weight, its completion's advantage: the reward arrives only at the end, and REINFORCE has no way to tell which tokens deserved it.
2. `kl_coef * kl_penalty(...)`: a leash to the **reference model** π_ref, a frozen copy of the starting model. Without it, the policy is free to drift into strange text that happens to score well ("reward hacking") or to forget how to write. The estimator `exp(d) - d - 1` (with d = log π_ref − log π) is always ≥ 0 and is 0 exactly where the two models agree.

The sum is divided by the total number of completion tokens in the batch, so the step size doesn't depend on how long the completions are.
"""),
    code(r"""
viz.show_code(reinforce.kl_penalty)
viz.show_code(reinforce.policy_loss)
"""),
    md(r"""
Compare with notebook 1's loss. Same core, two additions: the per-token mask, and the KL term.
"""),
    code(r"""
bandit_loss = '''def reinforce_loss(log_probs, rewards):
    return -(rewards.detach() * log_probs).mean()
'''
viz.show_diff(bandit_loss, reinforce.policy_loss, "notebook 1 (bandit)", "REINFORCE for an LLM")
"""),
    md(r"""
**The training loop.** Sample → score → advantages → reference log-probs → one gradient step (in micro-batches of 8 completions so memory stays bounded) → log. Then the samples are thrown away: after the update they no longer come from the current model.
"""),
    code(r"""
viz.show_code(reinforce.train)
"""),
    md(r"""
## Run it

Two copies of the model: the policy (trained, float32) and the reference (frozen, bfloat16). The plot updates live: watch **reward** (training accuracy), **format rate** (has a valid `<answer>` tag), **completion length** and **KL** (how far the policy has moved from the reference).
"""),
    code(r"""
tok = llm.load_tokenizer()
model = llm.load_model(trainable=True)
ref_model = llm.load_model(trainable=False)
cfg = llm.maybe_smoke(reinforce.Config())  # maybe_smoke only shrinks the automated CPU test; no effect on GPU
print(cfg)
"""),
    code(r"""
import time
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
live = viz.LivePlot(["reward", "format_rate", "completion_tokens", "kl"],
                    ["reward (train accuracy)", "well-formed answer rate", "completion length (tokens)", "KL to reference"],
                    every=2, color=viz.ALGO_COLORS["reinforce"])
t0 = time.time()
history, samples = reinforce.train(model, ref_model, tok, cfg, callback=live.update)
live.update(history, final=True)
minutes = (time.time() - t0) / 60
print(f"{minutes:.1f} minutes, peak GPU memory {llm.peak_memory_gb():.1f} GB")
"""),
    md(r"""
## The baseline at work

The running baseline trails the batch reward. Completions only get a positive advantage when they beat it, so as the model improves, "good enough" keeps getting harder.
"""),
    code(r"""
steps = [h["step"] for h in history]
plt.figure(figsize=(7, 3))
plt.plot(steps, [h["reward"] for h in history], color=viz.ALGO_COLORS["reinforce"], alpha=0.5, label="batch mean reward")
plt.plot(steps, [h["baseline"] for h in history], color="black", lw=2, label="running baseline")
plt.xlabel("step"), plt.legend(), plt.title("Reward vs. baseline"), plt.show()
"""),
    md(r"""
## What did training change, token by token?

Take one fixed completion and compare log π(token) under the trained model and under the starting model. Green tokens became more likely, red less likely. This is where you can see *what* got reinforced: usually the answer format and the arithmetic steps.
"""),
    code(r"""
torch.manual_seed(1)
probe = llm.generate(model, tok, task.eval_problems()[:4], samples_per_prompt=1, max_new_tokens=cfg.max_new_tokens)
lp_new = llm.batched_logprobs(model, probe)
lp_old = llm.batched_logprobs(ref_model, probe)
for i in range(2):
    tokens, pos = viz.completion_tokens(tok, probe, i)
    delta = (lp_new[i, pos] - lp_old[i, pos]).tolist()
    viz.show_tokens(tokens, delta, title=f"{probe.problems[i].question}  (reward {probe.rewards[i]:.0f}): Δ log π, trained − start")
"""),
    md(r"""
And the credit REINFORCE *assigns* during training: the completions for one prompt at the last step, each labelled with the advantage it was trained with. Every token inside a completion got that same weight. That's REINFORCE's crude credit assignment: one number per completion. PPO (notebook 5) gives each token its own.
"""),
    code(r"""
last = samples[-1]
viz.show_completions(last["completions"], title=f"Last training step: {last['question']}")
"""),
    md(r"""
## Before vs after

Same question, early in training versus the end:
"""),
    code(r"""
for s in (samples[0], samples[-1]):
    viz.show_completions(s["completions"][:2], title=f"step {s['step']}: {s['question']}")
"""),
    code(r"""
results = llm.evaluate(model, tok)
results.update(minutes=minutes, peak_memory_gb=llm.peak_memory_gb())
base = llm.load_run("base")
llm.save_run("reinforce", cfg, history, results, samples)

labels, accs = ["start", "REINFORCE"], [base["results"]["accuracy"] if base else float("nan"), results["accuracy"]]
plt.figure(figsize=(4, 2.8))
plt.bar(labels, accs, color=[viz.ALGO_COLORS["base"], viz.ALGO_COLORS["reinforce"]])
for i, a in enumerate(accs):
    plt.text(i, a + 0.01, f"{a:.0%}", ha="center")
plt.ylabel("held-out accuracy (greedy)"), plt.ylim(0, 1), plt.show()
"""),
    md(r"""
## What to take away

- REINFORCE on an LLM is notebook 1's loss applied to every token, plus a KL leash to the starting model.
- Every token in a completion shares its completion's advantage. The algorithm can't tell which step made the answer right.
- Each batch is used for exactly one gradient step, then discarded. Generation is the expensive part, so that's wasteful; GRPO fixes it next.
- If reward rises but the KL explodes or completions get weird, the policy is drifting: raise `kl_coef` or lower `lr`.
"""),
]

CELLS.append(md(r"""
## Further reading

- R. J. Williams (1992), [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696): REINFORCE.
- A. Ahmadian et al. (2024), [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740): REINFORCE and RLOO for LLMs.
- J. Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html): where the `exp(d) - d - 1` KL estimator comes from.
"""))
