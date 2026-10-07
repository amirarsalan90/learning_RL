"""Notebook 2: an LLM is a policy. Sampling, token log-probs, the reward, and the starting accuracy."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 2 · An LLM is a policy

**Run-only. Needs the GPU; takes a few minutes.** No training yet. This notebook looks at the pieces every algorithm uses:

1. what an "action" is for an LLM, and its log-probability;
2. the task and the reward function;
3. how good the starting model is, which decides how much there is to learn.

The model is `Qwen2.5-0.5B-Instruct`: small enough to fully fine-tune on a single 24 GB GPU such as an RTX 4090, big enough to do some arithmetic.
"""),
    code(r"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from rlcourse import llm, task, viz

print("device:", llm.DEVICE, "| model:", llm.MODEL_NAME)
tok = llm.load_tokenizer()
model = llm.load_model(trainable=False)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, vocabulary of {len(tok):,} tokens")
"""),
    code(r"""
viz.figure("llm_policy.svg")
"""),
    md(r"""
## The task and the prompt

Arithmetic like `What is 47 * 6 + 31?`. The model has to finish with `<answer>NUMBER</answer>`. Here's the exact text the model sees, after the chat template wraps it:
"""),
    code(r"""
problem = task.Problem("What is 47 * 6 + 31?", 47 * 6 + 31)
prompt = llm.chat_prompt(tok, problem.question)
print(prompt)
"""),
    md(r"""
The reward function is 15 lines of plain Python (`rlcourse/task.py`): find the last `<answer>…</answer>`, compare the number. That's the whole "environment".
"""),
    code(r"""
viz.show_code(task.parse_answer)
viz.show_code(task.reward)
for text in ["... so the answer is <answer>313</answer>", "<answer>312</answer>", "The answer is 313."]:
    print(f"{task.reward(text, problem.answer):.0f}  ←  {text!r}")
"""),
    md(r"""
## Sampling: one prompt, several completions

Every completion is one "action". Sampling uses the model's raw softmax at temperature 1 (no top-k or top-p), because the policy-gradient math assumes completions come from exactly the distribution whose log-probs we compute.
"""),
    code(r"""
torch.manual_seed(0)
r = llm.generate(model, tok, [problem], samples_per_prompt=6, max_new_tokens=200)
viz.show_completions([{"text": t, "reward": rw.item()} for t, rw in zip(r.texts, r.rewards)],
                     title=f"6 samples for: {problem.question}  (correct answer {problem.answer})")
"""),
    md(r"""
## Token log-probabilities

For training we need $\log \pi(\text{token}_t \mid \text{everything before it})$ for every completion token. One forward pass over prompt + completion gives all of them at once (`llm.token_logprobs`). The prompt's positions are masked out: the model didn't choose those tokens.

Below, each token of the first sample is shaded by how *surprised* the model was to produce it (red = low probability). Hover a token to see its log-prob.
"""),
    code(r"""
logp = llm.token_logprobs(model, r.input_ids, r.attention_mask)
tokens, pos = viz.completion_tokens(tok, r, 0)
viz.show_tokens(tokens, logp[0, pos].tolist(), title="log π(token) for each token of sample 1",
                caption="Darker red = less likely. The digits of a computation are often where the model hesitates.")
seq_logp = (logp * r.completion_mask).sum(1)
print("log π(whole completion) = sum of its token log-probs:")
for i in range(len(r)):
    print(f"  sample {i + 1}: {seq_logp[i].item():8.1f}   ({int(r.completion_mask[i].sum())} tokens, reward {r.rewards[i]:.0f})")
"""),
    md(r"""
Let's open up one decision. Here's the model's full next-token distribution at the point where it writes the first character of its final answer: the "arm" it's choosing among. Temperature reshapes this distribution, and so changes how much the model explores.
"""),
    code(r"""
row = 0
ids = r.input_ids[row]
comp_pos = r.completion_mask[row].nonzero().squeeze(-1).tolist()
text_so_far = ""
decision = comp_pos[0]
for p in comp_pos:  # find the token right after "<answer>"
    if text_so_far.rstrip().endswith("<answer>"):
        decision = p
        break
    text_so_far += tok.decode([ids[p]])
with torch.no_grad(), llm.autocast():
    am = r.attention_mask[row:row + 1, :decision]
    logits = model(input_ids=ids[None, :decision], attention_mask=am, position_ids=llm.position_ids(am)).logits[0, -1].float()

fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharey=True)
for ax, T in zip(axes, [0.5, 1.0, 1.5]):
    probs = torch.softmax(logits / T, -1)
    top = probs.topk(10)
    labels = [repr(tok.decode([i]))[1:-1] for i in top.indices.tolist()]
    ax.bar(range(10), top.values.cpu(), color=viz.PALETTE["blue"])
    ax.set_xticks(range(10)), ax.set_xticklabels(labels, rotation=45, fontsize=9)
    ax.set_title(f"temperature {T}: top token {top.values[0]:.0%}")
axes[0].set_ylabel("probability")
fig.suptitle(f"Next-token distribution after: …{tok.decode(ids[max(0, decision - 12):decision])!r}")
fig.tight_layout(), plt.show()
"""),
    md(r"""
## How good is the starting model?

RL can only reinforce what the model already *sometimes* does. From notebook 1: if every sample for a prompt gets the same reward, there's nothing to learn from it. So for each difficulty level we sample 8 completions for 32 problems and count how many of the 8 are right.
"""),
    code(r"""
n_problems, k = (8, 4) if llm.SMOKE else (32, 8)
per_problem = {}
for difficulty in ["easy", "medium", "hard"]:
    problems = task.TrainStream(difficulty, seed=1).next(n_problems)
    rr = llm.generate(model, tok, problems, samples_per_prompt=k, max_new_tokens=200)
    per_problem[difficulty] = rr.rewards.view(n_problems, k).sum(1).numpy()

fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), sharey=True)
for ax, (difficulty, counts) in zip(axes, per_problem.items()):
    hist = np.bincount(counts.astype(int), minlength=k + 1)
    colors = [viz.PALETTE["red"] if i in (0, k) else viz.PALETTE["green"] for i in range(k + 1)]
    ax.bar(range(k + 1), hist, color=colors)
    useful = ((counts > 0) & (counts < k)).mean()
    ax.set_title(f"{difficulty}: accuracy {counts.mean() / k:.0%}, useful prompts {useful:.0%}")
    ax.set_xlabel(f"correct out of {k} samples")
axes[0].set_ylabel("number of prompts")
fig.suptitle("Red bars: all right or all wrong, so zero learning signal for GRPO"), fig.tight_layout(), plt.show()
"""),
    md(r"""
**Pick the difficulty where most prompts are green**: the model is right sometimes but not always. The training notebooks default to `"medium"`. If `medium` is mostly red at 0 (too hard), switch the configs to `"easy"`; if mostly red at 8, use `"hard"`.

## Baseline: greedy accuracy on the held-out set

Every later notebook ends with the same evaluation, so we record the starting point: 200 held-out problems, greedy decoding (always the most likely token), accuracy and how often the answer is well-formed.
"""),
    code(r"""
results = llm.evaluate(model, tok)
print(f"accuracy {results['accuracy']:.1%}   well-formed <answer> tag {results['format_rate']:.1%}")
llm.save_run("base", None, [], results, [])
viz.show_completions([{"text": e["completion"], "reward": e["reward"], "note": e["question"]} for e in results["examples"][:4]],
                     title="Greedy answers from the starting model")
"""),
    md(r"""
## What to take away

- An LLM "action" is a whole completion; its log-prob is the sum of its tokens' log-probs, computed in one forward pass.
- The reward is just a function of the text. No gradient flows through it, or through sampling.
- Temperature controls exploration. RL needs some: if every sample is identical, every advantage is zero.
- RL can only sharpen behavior that the model sometimes shows already. The histogram above is the most useful thing to look at before any RL run.

Next: notebook 3 turns this into a training loop with REINFORCE.
"""),
]
