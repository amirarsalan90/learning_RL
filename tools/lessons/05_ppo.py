"""Notebook 5: PPO with a learned critic."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 5 · PPO: a critic that scores every token

**Run-only. Needs the GPU; training takes roughly 30–50 minutes.**

![PPO, animated](figures/ppo.gif)

PPO is the algorithm behind the original ChatGPT RLHF. It shares GRPO's clipped update, but replaces the group baseline with a **critic**: a second network that reads the completion as it's being written and predicts, at every token, the reward it expects at the end. That buys two things and costs one:

- ✅ **per-token credit**: tokens where the critic's prediction jumps up or down get the credit or blame, instead of every token sharing one number;
- ✅ **no groups needed**: one sample per prompt is enough, since the baseline comes from the critic;
- ❌ **a second model to train and keep in memory**, and one that has to learn before its advantages mean anything.
"""),
    code(r"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from rlcourse import llm, task, viz, grpo, ppo

viz.figure("ppo.svg")
"""),
    md(r"""
## The critic

Same transformer body as the policy, but instead of a 151k-way vocabulary head it has a head that outputs one number per position. `values[:, t]` is the critic's estimate of the final reward *before* token t is written. The head starts at zero, so at first the critic predicts 0 everywhere and has to learn from the returns.
"""),
    code(r"""
viz.show_code(ppo.Critic)
"""),
    md(r"""
## Per-token rewards and GAE

In classic RLHF-style PPO, the KL penalty isn't added to the loss (as in REINFORCE and GRPO); it's **subtracted from the reward at every token**. The real reward (0 or 1) lands on the last token.
"""),
    code(r"""
viz.show_code(ppo.token_rewards)
"""),
    md(r"""
Then **GAE** turns rewards and critic predictions into a per-token advantage. The building block is the *TD error*:

$$\delta_t = r_t + V_{t+1} - V_t$$

"How much better did things look after this token than before it?" If the critic thought the answer was 60% likely to be right, and after the model writes `=82` (a wrong product) it drops to 30%, that token gets δ ≈ −0.3. GAE then sums the upcoming δ's with a decay λ:

$$A_t = \delta_t + \lambda\, \delta_{t+1} + \lambda^2 \delta_{t+2} + \dots$$
"""),
    code(r"""
viz.show_code(ppo.gae)
"""),
    md(r"""
Here's GAE on a made-up 8-token completion that ends up **wrong** (reward 0), with a made-up critic that gets pessimistic at token 3. λ controls how far blame spreads back: λ = 0 blames mostly the token where the prediction dropped. λ = 1 skips the critic's later predictions: each token gets (final reward − the critic's prediction before it), so the blame is spread over every token, much like REINFORCE with the critic as its baseline.
"""),
    code(r"""
values = torch.tensor([[0.0, 0.6, 0.62, 0.6, 0.25, 0.22, 0.2, 0.2, 0.15]])  # V before each token (pos 0 is the prompt)
mask = torch.tensor([[0.0, 1, 1, 1, 1, 1, 1, 1, 1]])
rewards = torch.zeros_like(values)
rewards[0, -1] = 0.0  # wrong answer
labels = ["", "23", "*4", "=82", "+17", "=99", "<ans", "99>", "<end>"]
fig, axes = plt.subplots(1, 4, figsize=(15, 2.8), sharey=True)
axes[0].bar(range(9), values[0], color=viz.PALETTE["purple"])
axes[0].set_title("critic V before each token")
for ax, lam in zip(axes[1:], [0.0, 0.95, 1.0]):
    cfg_demo = ppo.Config(lam=lam)
    adv, _ = ppo.gae(rewards, values, mask, cfg_demo)
    a = adv[0].numpy()
    ax.bar(range(9), a, color=[viz.PALETTE["green"] if x > 0 else viz.PALETTE["red"] for x in a])
    ax.set_title(f"GAE advantage, λ = {lam}")
for ax in axes:
    ax.set_xticks(range(9)), ax.set_xticklabels(labels, rotation=45, fontsize=8), ax.axhline(0, color="black", lw=0.6)
fig.tight_layout(), plt.show()
"""),
    md(r"""
**Notice:** at λ = 0 almost all the blame lands on `=82`, the token where things went wrong. That's the promise of a critic. The catch: the critic has to be *good* for this to help, and early in training it isn't. We use λ = 0.95, a common compromise.

## The PPO losses

The policy loss is GRPO's clipped loss, except the advantage now has one value per token (shape `(batch, seq_len)` rather than `(batch,)`), and the KL lives in the reward instead. The critic is trained by regression toward the returns (advantage + old value), also with a clip.
"""),
    code(r"""
viz.show_diff(grpo.policy_loss, ppo.policy_loss, "GRPO policy_loss", "PPO policy_loss")
viz.show_code(ppo.value_loss)
"""),
    md(r"""
## Run it

Same total completions per step as GRPO (64), but now 64 different prompts with one sample each. Watch **explained variance**: how well the critic predicts the returns (0 = no better than a constant, 1 = perfect). The advantages only become meaningful once it rises above 0. **Value at start** is the critic's guess of P(correct) right after reading the prompt; it should track the actual reward.
"""),
    code(r"""
tok = llm.load_tokenizer()
model = llm.load_model(trainable=True)
ref_model = llm.load_model(trainable=False)
critic = ppo.Critic().to(llm.DEVICE)
cfg = llm.maybe_smoke(ppo.Config())
print(cfg)
"""),
    code(r"""
import time
torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
live = viz.LivePlot(["reward", "kl", "explained_variance", "value_at_start", "value_loss", "clip_frac"],
                    ["reward (train accuracy)", "KL to reference", "critic explained variance",
                     "critic: predicted P(correct) at start", "value loss", "clip fraction"],
                    every=2, color=viz.ALGO_COLORS["ppo"])
t0 = time.time()
history, samples = ppo.train(model, ref_model, critic, tok, cfg, callback=live.update)
live.update(history, final=True)
minutes = (time.time() - t0) / 60
print(f"{minutes:.1f} minutes, peak GPU memory {llm.peak_memory_gb():.1f} GB")
"""),
    md(r"""
## Watch the critic read

The trained critic, run over a few fresh completions: its predicted chance of success after each token. Look for where the line jumps: that's where the critic "noticed" something, like a wrong intermediate result or a well-formed answer tag.
"""),
    code(r"""
torch.manual_seed(3)
probe = llm.generate(model, tok, task.eval_problems()[:8], samples_per_prompt=1, max_new_tokens=cfg.max_new_tokens)
with torch.no_grad():
    v = critic(probe.input_ids, probe.attention_mask)
picks = [i for i in range(len(probe)) if probe.rewards[i] == 1][:1] + [i for i in range(len(probe)) if probe.rewards[i] == 0][:1]
fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.6 * len(picks)), squeeze=False)
for ax, i in zip(axes[:, 0], picks):
    tokens, pos = viz.completion_tokens(tok, probe, i)
    ax.plot(range(len(pos)), v[i, pos].cpu(), "o-", ms=3, color=viz.PALETTE["purple"])
    ax.axhline(probe.rewards[i], ls="--", color=viz.PALETTE["green"] if probe.rewards[i] else viz.PALETTE["red"],
               label=f"actual reward {probe.rewards[i]:.0f}")
    step = max(1, len(tokens) // 40)
    ax.set_xticks(range(0, len(tokens), step))
    ax.set_xticklabels([repr(t)[1:-1][:8] for t in tokens[::step]], rotation=70, fontsize=7)
    ax.set_ylabel("V"), ax.set_title(probe.problems[i].question, fontsize=10), ax.legend(fontsize=8)
fig.tight_layout(), plt.show()
"""),
    md(r"""
And the per-token advantages those values produce for the same completions. Unlike REINFORCE and GRPO, the shading varies **within** a completion.
"""),
    code(r"""
with torch.no_grad():
    old_logp = llm.batched_logprobs(model, probe)
    ref_logp = llm.batched_logprobs(ref_model, probe)
mask = probe.completion_mask.float()
tr = ppo.token_rewards(probe.rewards.to(llm.DEVICE), old_logp, ref_logp, mask, cfg)
adv, _ = ppo.gae(tr, v * mask, mask, cfg)
for i in picks:
    tokens, pos = viz.completion_tokens(tok, probe, i)
    viz.show_tokens(tokens, adv[i, pos].tolist(), title=f"Per-token advantage (reward {probe.rewards[i]:.0f}): {probe.problems[i].question}")
"""),
    code(r"""
results = llm.evaluate(model, tok)
results.update(minutes=minutes, peak_memory_gb=llm.peak_memory_gb())
llm.save_run("ppo", cfg, history, results, samples)

runs = {n: llm.load_run(n)["history"] for n in ["reinforce", "grpo"] if llm.load_run(n)}
runs["ppo"] = history
viz.compare_runs(runs, ["reward", "kl"], ["reward (train accuracy)", "KL to reference"], x="tokens")
plt.show()
names = [n for n in ["base", "reinforce", "grpo", "ppo"] if llm.load_run(n)]
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
for ax, key, label in [(axes[0], "accuracy", "held-out accuracy"), (axes[1], "peak_memory_gb", "peak GPU memory (GB)")]:
    shown = [n for n in names if key in llm.load_run(n)["results"]]
    ax.bar(shown, [llm.load_run(n)["results"][key] for n in shown], color=[viz.ALGO_COLORS[n] for n in shown])
    ax.set_title(label)
fig.tight_layout(), plt.show()
"""),
    md(r"""
## What to take away

- PPO's critic predicts the final reward at every token; GAE turns changes in that prediction into **per-token advantages**.
- The clipped policy loss is the same as GRPO's; what differs is where the baseline and the KL come from.
- The critic is a whole second model to train: about double the memory, and its advantages are noise until its explained variance rises.
- That cost, for tasks where the reward only arrives at the end anyway, is why GRPO became popular for reasoning models.
"""),
]

CELLS.append(md(r"""
## Further reading

- J. Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): PPO.
- J. Schulman et al. (2015), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): GAE.
- L. Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT): PPO for RLHF, with the KL penalty in the reward.
"""))
