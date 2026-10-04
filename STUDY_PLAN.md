# Study plan: learn RL by implementing it

## How to use this repository

Run each complete example, read its notebook alongside the implementation, and
predict the result of an ablation before running it. Advance when you can explain
the gradients and failure cases, rather than when a fixed number of days has passed.

Each new lesson gets a CPU/small-batch smoke test before longer experiments.
For the LLM lessons, retain a shared task and evaluator so changing algorithms
does not also change what success means.

## 1. REINFORCE and baselines — available

**Model/environment:** three contexts, three actions, stochastic Bernoulli rewards;
contexts are uniformly sampled. Each context has a different optimal action and
expected reward. A softmax table is the policy; a scalar per context is the critic.

**Run:** `python scripts/run_bandit.py` and `notebooks/01_reinforce_bandit.ipynb`.

**Experiments:** no baseline; a running mean of previous rewards; a learned value
per context. Five seeds, 1,000 updates, 64 samples/update, SGD learning rate 0.5.
Compare expected reward, optimal-action probability, entropy, and estimator variance.
At a fixed uniform policy, enumerate the exact gradient and variance and compare
them with Monte Carlo estimates using identical batches for every baseline.

**Understand before advancing:**

- Why is the gradient `(reward - baseline) * grad(log probability)`?
- Why can a context-dependent baseline leave its expectation unchanged?
- Why must the policy use a detached baseline from before the current update?
- Why does value regression differ from policy optimization?
- Why can lower variance fail to produce a visibly better final learning curve?

**Suggested follow-up:** change the learning rate to 0.1 and 1.0; change the batch
size to 8 and 128. Compare methods at equal sampled-action budgets, not just equal
step counts. Write a short explanation of your observations before lesson 2.

## 2. LLM evaluation and SFT — planned

**Default:** Qwen/Qwen2.5-0.5B-Instruct; BF16, LoRA, short responses. First confirm
CUDA access and run a memory smoke test. Install and pin a compatible Transformers,
TRL, PEFT, Datasets, and Accelerate stack at this milestone, separately from the
introductory environment requirements.

Generate arithmetic problems with a strict `<answer>integer</answer>` parser and
reward 1 only for a well-formed, correct answer. Split by underlying problem;
exclude duplicate/rephrased evaluation problems from training. Calibrate task
difficulty using sampled completions so the policy sometimes succeeds and fails.
Build a small SFT baseline from correct solutions.

**Success:** reproducible starting accuracy; tested parsing and completion masks;
saved examples and a disjoint evaluation set. Track correctness independently of
format validity. Do not interpret improved arithmetic reward as proof of general
reasoning improvement.

## 3. LLM REINFORCE — planned

Implement generation, deterministic reward, recomputation of completion-token log
probabilities with gradients, and fresh-rollout policy updates. Mask prompt,
padding, and tokens after EOS. Start with summed completion log probabilities;
inspect how length weighting changes when averaging tokens instead.

Compare no baseline, previous running reward mean, and explicit reference-policy
regularization. Save examples alongside correctness, entropy, length, generated
tokens, wall time, and peak memory. Verify the update sign on a tiny batch first.

**Success:** explain why generation/reward need no gradients and trace one sampled
completion all the way through its scalar reward to its token gradients.

## 4. GRPO — planned

Use TRL with explicitly recorded loss type, normalization, KL coefficient, clipping,
and rollout reuse settings. Inspect actual rewards and advantages for each prompt.
Compare group size 4 versus 8 at matched generated-token budgets, mean-centering
versus standard-deviation scaling, and one versus multiple updates per rollout.

**Success:** explain zero-variance groups, the old-policy probability ratio, when
clipping becomes active, and how the group baseline replaces a learned critic.
Distinguish the old rollout policy from the frozen KL reference policy.

## 5. DPO — planned

Generate a fixed response pool from the same initial checkpoint. Make chosen/rejected
pairs only where a prompt has both a successful and unsuccessful response. Train
offline with TRL, then evaluate on the same held-out tasks used for online methods.
Inspect chosen/rejected log probabilities relative to the reference policy.

**Success:** explain the preference loss, why it requires no online rollout/reward
loop, and how limited pair coverage restricts what can be learned.

## 6. PPO — planned

Use the same deterministic reward, adding a value model, returns/advantages, and
clipped policy updates. Select and pin a compatible PPO implementation when this
lesson is built. Inspect critic predictions, policy ratios, and separate losses.
A learned reward model is an optional later extension, not a prerequisite.

**Success:** explain what PPO adds to REINFORCE/GRPO, how value estimates affect
advantages, and why the old policy and reference policy have different roles.

## Comparison discipline

Reset each LLM method to the same initial checkpoint. Keep prompts, evaluation,
decoding settings, and reward definitions controlled. Report generated-token
budgets and runtime as well as updates; offline and online methods incur different
data-generation costs. Track seeds and exact installed package versions.

Learning curves are evidence about these experiments, not a universal ranking of
algorithms. Read sampled failures whenever reward increases unexpectedly.
