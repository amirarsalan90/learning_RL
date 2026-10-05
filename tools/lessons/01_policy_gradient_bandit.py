"""Stage 1: policy gradients from zero, on a 5-armed bandit."""


def md(text):
    return ("md", text, None)


def code(text):
    return ("code", text, None)


def think(question, answer):
    return ("think", question, answer)


CELLS = [
    md(r"""
# Stage 1 · Policy gradients from zero

**Goal:** understand the one idea that REINFORCE, PPO and GRPO are all built on, by implementing it yourself on the smallest problem that still has it. Everything here runs on a CPU in seconds.

### Why start with a bandit?

RL for LLMs, with the details stripped away, is a loop:

1. Give the model a prompt.
2. It **samples** a completion.
3. A reward function **scores** the completion (say, 1 if the math answer is right, else 0).
4. **Nudge** the model so that high-reward completions become more likely.

A *bandit* is that loop with everything else removed:

| RL for LLMs | This notebook |
|---|---|
| the policy is the LLM | the policy is a softmax over 5 learnable numbers |
| an action is a whole completion | an action is one of 5 "arms" |
| reward from a verifier (answer correct → 1) | reward is a coin flip; each arm has its own hidden P(reward = 1) |
| log π(completion) = Σ log π(token) | log π(arm) |
| impossible to enumerate every completion | only 5 arms, so we **can** enumerate them |

The last row is why we start here. Because we can enumerate everything, we can compute the *exact* gradient and check whether our sampling-based estimate gets it right. With an LLM you never get to see the true answer.

### How to work through this

- Cells with `# YOUR CODE HERE` are yours. Delete the `raise NotImplementedError` and write the code. Each is followed by a check cell.
- **Think** questions have an *Your answer:* cell under them. Write a sentence or two before you run the next cell; predicting first is where most of the learning happens.
- Stuck for more than ~15 minutes? Peek at `solutions/01_policy_gradient_bandit.ipynb`, or ask in the project thread.
"""),
    code(r"""
import torch
import matplotlib.pyplot as plt
from rlcourse import checks

torch.manual_seed(0)
"""),
    md(r"""
## 1 · The environment

Five arms. Pulling arm $a$ pays reward 1 with probability $p_a$, else 0. The agent never sees `TRUE_P`; it only sees the rewards of the arms it pulls.

Think of reward 1 as "the answer was correct". Like an LLM verifier, `pull` is a **black box**: plain Python, no gradients flow through it.
"""),
    code(r"""
TRUE_P = torch.tensor([0.20, 0.50, 0.80, 0.35, 0.65])  # hidden from the agent; arm 2 is best
N_ARMS = len(TRUE_P)


def pull(actions):
    # Environment: a tensor of arm indices in, a tensor of 0/1 rewards out.
    return torch.bernoulli(TRUE_P[actions])


pull(torch.tensor([2, 2, 2, 2, 0, 0, 0, 0]))
"""),
    md(r"""
## 2 · The policy

The parameters $\theta$ are 5 logits, and the policy is $\pi_\theta(a) = \mathrm{softmax}(\theta)_a$. We start at $\theta = 0$, a uniform policy.

An LLM is exactly this, except the logits come out of a transformer (one softmax over the vocabulary per token) instead of being free parameters.

### Exercise 2.1: sample actions and their log-probabilities

Implement `sample(logits, n)`, returning

- `actions`: shape `(n,)`, integer arm indices drawn from $\pi_\theta$;
- `log_probs`: shape `(n,)`, where `log_probs[i]` $= \log \pi_\theta(a_i)$, **differentiable** with respect to `logits`.

Useful: `torch.softmax`, `torch.log_softmax`, `torch.multinomial(probs, n, replacement=True)`, tensor indexing.

The sampling itself is not differentiable (you can't take the derivative of "which arm came up"), which is fine; we only need gradients of the log-probabilities. That turns out to be the whole trick.

Why log-probs rather than probs? For an LLM, the probability of a completion is a product of hundreds of token probabilities, which underflows. Its log is a sum, which is stable. Everything in policy-gradient land is written in log-probs.
"""),
    code(r"""
def sample(logits, n):
    ### BEGIN SOLUTION
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs.detach(), n, replacement=True)
    log_probs = torch.log_softmax(logits, dim=-1)[actions]
    ### END SOLUTION
    return actions, log_probs
"""),
    code(r"""
checks.check_sample(sample)
"""),
    md(r"""
## 3 · The objective, and the gradient we wish we had

We want to maximize the expected reward

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[r] = \sum_a \pi_\theta(a)\, p_a .$$

For this tiny bandit, and *only because we're cheating and know* $p$, we can write $J$ down and let autograd compute the exact gradient $\nabla_\theta J$. That's our ground truth.

### Exercise 3.1: the exact expected reward
"""),
    code(r"""
def expected_reward(logits):
    ### BEGIN SOLUTION
    return (torch.softmax(logits, dim=-1) * TRUE_P).sum()
    ### END SOLUTION
"""),
    code(r"""
checks.check_expected_reward(expected_reward, TRUE_P)

logits = torch.zeros(N_ARMS, requires_grad=True)
J = expected_reward(logits)
J.backward()
exact_grad = logits.grad.clone()
print(f"J at the uniform policy = {J.item():.3f}")
print("exact gradient         =", exact_grad.numpy().round(4))
"""),
    think(
        r"""
**Think 3.2.** Which arms get a positive gradient, which get a negative one, and what decides it? (Compare each $p_a$ with $J$.)
""",
        r"""
For a softmax policy, $\partial J / \partial \theta_a = \pi(a)\,(p_a - J)$. Arms that are better than the policy's current average (arms 2 and 4, with $p > 0.5$) are pushed up; worse-than-average arms are pushed down; arm 1 sits exactly at the average and gets 0. "Better than the current average" is precisely the idea of an **advantage**, which comes back in section 6 and in every algorithm after this.
""",
    ),
    md(r"""
**Why can't we do this for an LLM?** Two reasons. We can't sum over every possible completion (there are vocabulary-size-to-the-power-of-length of them), and we don't know $p$: the reward is a black box we can only query on samples we actually generate. All we ever have is: samples, their rewards, and the ability to differentiate $\log \pi$ of those samples.

## 4 · The log-derivative trick (REINFORCE)

Here is the derivation. It's three lines and it is the most important thing in this course:

$$
\begin{aligned}
\nabla_\theta J
&= \sum_a p_a \,\nabla_\theta \pi_\theta(a) \\
&= \sum_a p_a \,\pi_\theta(a)\,\nabla_\theta \log \pi_\theta(a) \qquad \text{because } \nabla \log \pi = \tfrac{\nabla \pi}{\pi}\\
&= \mathbb{E}_{a\sim\pi_\theta}\!\left[\, p_a \,\nabla_\theta \log \pi_\theta(a) \right]
\;\approx\; \frac1N \sum_{i=1}^N r_i\, \nabla_\theta \log \pi_\theta(a_i).
\end{aligned}
$$

The last step replaces an expectation over $\pi$ with an average over samples from $\pi$, and the unknown $p_{a_i}$ with the observed reward $r_i$ (an unbiased draw of it). Everything left is computable: samples, rewards, and gradients of log-probs. We never differentiate through the sampling or the reward.

To make autograd compute this for us, we write a **surrogate loss** whose gradient is minus that estimate:

$$L(\theta) = -\frac1N \sum_i r_i \,\log \pi_\theta(a_i), \qquad r_i \text{ treated as a constant.}$$

### Exercise 4.1: the REINFORCE loss
"""),
    code(r"""
def reinforce_loss(log_probs, rewards):
    ### BEGIN SOLUTION
    return -(rewards.detach() * log_probs).mean()
    ### END SOLUTION
"""),
    code(r"""
checks.check_reinforce_loss(reinforce_loss)
"""),
    md(r"""
Now the moment of truth. Estimate the gradient from $N$ samples and compare it with the exact one. *Cosine* is the cosine similarity between the estimate and the exact gradient: 1 means it points in exactly the right direction.
"""),
    code(r"""
def estimate_grad(n, weight_fn=None, reward_offset=0.0):
    # One REINFORCE gradient estimate at the uniform policy, from n samples.
    # weight_fn turns rewards into the per-sample weights (we'll use it in section 6).
    logits = torch.zeros(N_ARMS, requires_grad=True)
    actions, log_probs = sample(logits, n)
    rewards = pull(actions) + reward_offset
    weights = rewards if weight_fn is None else weight_fn(rewards)
    reinforce_loss(log_probs, weights).backward()
    return -logits.grad  # minus: the loss gradient points downhill, we want uphill on J


cos = torch.nn.functional.cosine_similarity
print("exact       ", exact_grad.numpy().round(3))
for n in [8, 8, 8, 100, 10_000, 1_000_000]:
    g = estimate_grad(n)
    print(f"N={n:<10,}", g.numpy().round(3), f" cosine={cos(g, exact_grad, dim=0).item():+.2f}")
"""),
    md(r"""
With many samples the estimate converges to the exact gradient: the trick works. With 8 samples it's noisy, sometimes badly. That matters, because LLM RL lives at small $N$: GRPO typically samples 8 to 16 completions per prompt, and each one costs a full generation. **Variance is the central practical problem**, and section 6 is about reducing it.
"""),
    think(
        r"""
**Think 4.2.** The *value* of `reinforce_loss` is not $J$ and is nearly meaningless as a training curve; it can go up while the policy is getting better. Why? What would you log instead to see whether training works?
""",
        r"""
The surrogate only has the right *gradient* at the current parameters. Its value depends on the batch you happened to sample: as the policy improves, the log-probs of good actions approach 0 from below, so $-r\log\pi$ shrinks, but batches with more rewards also have more nonzero terms, and the value moves for reasons unrelated to progress. Log the **reward** (mean reward of the batch, or here the exact $J$), plus things like the probability of the best action and the policy's entropy. In LLM RL you log mean reward, accuracy on a held-out set, completion length and KL to the starting model, never the surrogate loss as a measure of progress.
""",
    ),
    think(
        r"""
**Think 4.3.** With 0/1 rewards, what gradient does a sample with $r = 0$ contribute? If failures contribute nothing, how do bad arms ever become *less* likely?
""",
        r"""
A sample with $r=0$ contributes exactly zero. Bad arms lose probability only indirectly: when a rewarded arm's logit goes up, the softmax renormalizes and every other arm's probability goes down. So learning comes entirely from successes, and if the policy almost never succeeds (an LLM on a task that's too hard), there is almost no learning signal. Keep this in mind for choosing task difficulty in stage 3.
""",
    ),
    md(r"""
## 5 · Train it

### Exercise 5.1: the training loop

Fill in one step of training. It's five lines:

1. `sample` a batch of `batch_size` actions from the current logits;
2. `pull` them, and add `reward_offset` to every reward (it's 0 for now; it's an experiment for section 6);
3. turn rewards into weights: `weights = rewards if weight_fn is None else weight_fn(rewards)` (also for section 6);
4. compute `reinforce_loss(log_probs, weights)`;
5. `opt.zero_grad()`, `loss.backward()`, `opt.step()`.

We use plain SGD, so each update is exactly `logits -= lr * grad`, with no momentum or Adam scaling to hide what's going on.
"""),
    code(r"""
def train(steps=300, batch_size=16, lr=0.5, weight_fn=None, reward_offset=0.0, seed=0):
    torch.manual_seed(seed)
    logits = torch.zeros(N_ARMS, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=lr)
    history = []
    for step in range(steps):
        ### BEGIN SOLUTION
        actions, log_probs = sample(logits, batch_size)
        rewards = pull(actions) + reward_offset
        weights = rewards if weight_fn is None else weight_fn(rewards)
        loss = reinforce_loss(log_probs, weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ### END SOLUTION
        with torch.no_grad():  # evaluation only, using the hidden TRUE_P
            history.append(expected_reward(logits).item())
    return logits.detach(), history
"""),
    code(r"""
checks.check_train(train)
"""),
    code(r"""
def plot_runs(runs, title):
    # runs: {label: [history, history, ...]}, one history per seed
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(7, 3.5))
    for color, (label, histories) in zip(colors, runs.items()):
        for i, h in enumerate(histories):
            plt.plot(h, color=color, alpha=0.7, lw=1, label=label if i == 0 else None)
    plt.axhline(TRUE_P.max(), ls="--", c="gray", lw=1, label="best possible")
    plt.xlabel("update step"), plt.ylabel("expected reward J"), plt.title(title)
    plt.ylim(0.15, 0.85), plt.legend(), plt.show()


final_logits, _ = train()
print("final policy:", torch.softmax(final_logits, -1).numpy().round(3))
plot_runs({"REINFORCE": [train(seed=s)[1] for s in range(5)]}, "5 seeds, batch 16, lr 0.5")
"""),
    think(
        r"""
**Think 5.2.** Before running anything, predict: what happens with `lr=0.05`? With `lr=5`? With `batch_size=2` versus `batch_size=128`? Then try them in the cell below and explain what you see.
""",
        r"""
`lr=0.05` learns slowly but smoothly. `lr=5` makes huge, noisy jumps: a few lucky early successes on a mediocre arm can push its logit so high that the policy stops sampling anything else and gets stuck there ("premature collapse"; entropy goes to 0 and exploration stops). `batch_size=2` gives very noisy gradients; `batch_size=128` gives smooth curves but costs 64x more samples per step. Compare at an equal *number of samples* (steps × batch size), not equal steps: in LLM RL, samples (generations) are the expensive part.
""",
    ),
    code(r"""
# Your experiments here, e.g.
# plot_runs({"lr 0.05": [train(lr=0.05, seed=s)[1] for s in range(5)],
#            "lr 5":    [train(lr=5.0, seed=s)[1] for s in range(5)]}, "learning rate")
"""),
    md(r"""
## 6 · Baselines: same gradient on average, much less noise

An experiment first. Give every reward a bonus of +5, "five points for showing up". This doesn't change which arm is best: the optimal policy is identical. Predict what happens to training, then run it.
"""),
    code(r"""
plot_runs({
    "rewards 0/1": [train(seed=s)[1] for s in range(5)],
    "rewards +5":  [train(reward_offset=5.0, seed=s)[1] for s in range(5)],
}, "same problem, rewards shifted by +5")
"""),
    md(r"""
Several of the shifted runs lock onto the wrong arm. Why? Now *every* sample has a large positive weight, so every arm you happen to sample gets pushed up hard. On average those pushes cancel out to the right gradient, but with 16 samples it's mostly "boost whatever got sampled". An unlucky early batch boosts a mediocre arm, which gets sampled more, which gets boosted more.

**The fix: subtract a baseline.** Use $(r_i - b)$ instead of $r_i$. This doesn't change the expected gradient, for any $b$ that doesn't depend on the action being scored:

$$\mathbb{E}_{a\sim\pi}\!\left[\, b\, \nabla \log \pi(a) \right] = b \sum_a \pi(a)\,\frac{\nabla \pi(a)}{\pi(a)} = b\, \nabla \sum_a \pi(a) = b\, \nabla 1 = 0 .$$

But it can cut the variance enormously. $A_i = r_i - b$ is called the **advantage**: how much better this sample did than expected. Positive advantage pushes the sample's probability up, negative pushes it down. Notice that bad samples now get pushed *down* directly, which answers Think 4.3.

### Exercise 6.1: batch-mean baseline

Use the mean reward of the batch as $b$. This is exactly what **GRPO** does: sample a group of completions for the same prompt, and each completion's advantage is its reward minus the group's mean reward. No learned critic needed.
"""),
    code(r"""
def mean_baseline(rewards):
    ### BEGIN SOLUTION
    return rewards - rewards.mean()
    ### END SOLUTION
"""),
    code(r"""
checks.check_mean_baseline(mean_baseline)
"""),
    md(r"""
### Exercise 6.2: normalized advantages

GRPO goes one step further and divides by the group's standard deviation: $A_i = (r_i - \text{mean}) / (\text{std} + \epsilon)$. Use `rewards.std()` and $\epsilon = 10^{-6}$.
"""),
    code(r"""
def normalized(rewards):
    ### BEGIN SOLUTION
    return (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    ### END SOLUTION
"""),
    code(r"""
checks.check_normalized(normalized)
"""),
    md(r"""
Now measure what each choice does to a single gradient estimate, at the uniform policy with batches of 16, averaged over 2,000 batches:

- **avg cosine** with the exact gradient: how well *one* batch points the right way (higher is better);
- **scale**: the average estimate projected on the exact gradient, divided by its length. 1.0 means unbiased; anything else means the estimate is systematically scaled.
"""),
    code(r"""
def grad_stats(weight_fn, reward_offset=0.0, n=16, trials=2000):
    torch.manual_seed(0)
    grads = torch.stack([estimate_grad(n, weight_fn, reward_offset) for _ in range(trials)])
    avg_cos = cos(grads, exact_grad.expand_as(grads), dim=1).mean().item()
    scale = (grads.mean(0) @ exact_grad / exact_grad.norm() ** 2).item()
    return avg_cos, scale


print(f"{'weights':<16}{'offset':>7}{'avg cosine':>12}{'scale':>8}")
for label, fn in [("raw reward", None), ("mean baseline", mean_baseline), ("normalized", normalized)]:
    for offset in [0.0, 5.0]:
        c, s = grad_stats(fn, offset)
        print(f"{label:<16}{offset:>7}{c:>12.2f}{s:>8.2f}")
"""),
    code(r"""
plot_runs({
    "raw, +5":           [train(reward_offset=5.0, seed=s)[1] for s in range(5)],
    "mean baseline, +5": [train(reward_offset=5.0, weight_fn=mean_baseline, seed=s)[1] for s in range(5)],
    "normalized, +5":    [train(reward_offset=5.0, weight_fn=normalized, seed=s)[1] for s in range(5)],
}, "baselines make the offset irrelevant")
"""),
    think(
        r"""
**Think 6.3.** Read the table. (a) Why does the mean baseline make the offset irrelevant? (b) The mean baseline's scale is a bit below 1.0. Where does that small bias come from? (Hint: whose reward is inside the mean?) (c) Normalizing changes the scale a lot. Is that a problem? (d) Raw rewards are unbiased in theory, yet with +5 the measured scale isn't 1.0 either. Why not?
""",
        r"""
(a) Adding 5 to every reward also adds 5 to the mean, so $r_i - \bar r$ is unchanged: the offset cancels exactly. (b) Sample $i$'s own reward is part of $\bar r$, so the baseline *does* depend on its action. Algebra gives $r_i - \bar r = \tfrac{N-1}{N}\,(r_i - \text{mean of the others})$, so the expected gradient is the true one times $(N-1)/N = 15/16 \approx 0.94$. It points the right way, just slightly shorter. Exercise 6.5 removes it. (c) Not really: the direction is still right and the learning rate absorbs the scale. What normalizing buys is invariance to the reward's scale (rewards in 0/1 or 0/100 train the same), and it gives every prompt's group equal weight. One side effect, discussed in later papers (e.g. Dr. GRPO): prompts whose rewards barely vary get their small differences blown up to unit size. (d) The raw +5 estimator *is* unbiased, but its variance is so large that even the average of 2,000 batches is still noticeably off. That's the same noise that sent training to the wrong arm.
""",
    ),
    think(
        r"""
**Think 6.4.** With the mean baseline, what happens when every sample in a batch gets the same reward (all 0, or all 1)? What does that imply for an LLM trained with GRPO on prompts that are far too hard or far too easy?
""",
        r"""
All advantages are 0, so the batch contributes no gradient at all. For GRPO, a prompt the model always fails (or always solves) is wasted compute: you paid for generating the whole group and learned nothing from it. That's why the task's difficulty has to be calibrated so the model sometimes succeeds and sometimes fails, and why some GRPO variants filter out zero-variance groups (e.g. DAPO's dynamic sampling).
""",
    ),
    md(r"""
### Exercise 6.5 (optional): leave-one-out baseline

Remove the bias from 6.3(b): give sample $i$ the baseline $b_i = \frac{1}{N-1}\sum_{j\neq i} r_j$, the mean of the *other* rewards. This is the **RLOO** estimator (REINFORCE Leave-One-Out), which is used for LLM training too. Do it without a Python loop: the sum of the others is `rewards.sum() - rewards`.
"""),
    code(r"""
def loo_baseline(rewards):
    ### BEGIN SOLUTION
    n = len(rewards)
    return rewards - (rewards.sum() - rewards) / (n - 1)
    ### END SOLUTION
"""),
    code(r"""
checks.check_loo_baseline(loo_baseline)
c, s = grad_stats(loo_baseline)
print(f"leave-one-out: avg cosine {c:.2f}, scale {s:.2f}")
"""),
    md(r"""
## 7 · From this bandit to an LLM

You've now implemented the core of every algorithm on the roadmap. Here's how it maps:

| Here | With an LLM |
|---|---|
| `sample(logits, n)` | `model.generate(prompt, do_sample=True)` n times, then a forward pass to get each token's log-prob |
| $\log \pi(a)$ | $\sum_t \log \pi(y_t \mid \text{prompt}, y_{<t})$, summed over completion tokens only |
| `pull(actions)` | a reward function, e.g. parse the final answer and compare it with the right one |
| batch of samples | a *group* of completions for the same prompt |
| `mean_baseline` / `normalized` | GRPO's group-relative advantage |
| `reinforce_loss` | the same loss, with each token of a completion weighted by that completion's advantage |

What PPO and GRPO add on top, each of which we'll build when it becomes necessary:

- **Reusing samples.** Generation is expensive, so you'd like several gradient steps per batch. But after one step the samples come from an *old* policy; correcting for that requires a probability ratio $\pi_\theta / \pi_{\text{old}}$, and **clipping** that ratio keeps updates from running away (Think 5.2's `lr=5` collapse is what it prevents).
- **A KL leash** to the original model, so the policy can't drift into reward-hacking gibberish.
- **(PPO only) a learned value function** as the baseline instead of a group mean, which needs no group but costs a whole second network.

**Before stage 2:** make sure your *Your answer* cells are filled in, and post anything that surprised you or didn't click in the project thread. We'll go through it before moving on.
"""),
]
