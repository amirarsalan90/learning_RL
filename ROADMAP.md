# Roadmap

Each stage adds **one** new idea on top of the previous one, and each is small enough to run in minutes on a single RTX 4090. You write the core of every algorithm yourself in plain PyTorch, in notebooks with TODO blanks and checks; nothing is hidden inside a library until the very last stage, where you compare your code with one.

The LLM stages share one model and one task, so differences between algorithms aren't confused with differences in setup:

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, fully fine-tuned in bf16 (small enough that no LoRA or multi-GPU tricks are needed, so the training loop stays readable).
- **Task:** short arithmetic word problems with a checkable answer in `<answer>…</answer>`; reward 1 if the answer is right, else 0. Tuned so the starting model is right some of the time but not most of the time.

| # | Stage | The new idea | Runs on |
|---|---|---|---|
| 1 | **Policy gradients on a bandit** | The log-derivative trick, the surrogate loss, baselines and advantages, GRPO-style normalization | CPU, seconds |
| 2 | **A tiny sequence policy** | Actions become token *sequences*: summing token log-probs, EOS and masking, one reward for many tokens, length bias, entropy collapse | CPU or GPU, ~1 min |
| 3 | **An LLM as a policy** | Sampling from Qwen, computing completion log-probs yourself (and matching Hugging Face's), writing the reward function, measuring the starting accuracy. No training yet | 4090 |
| 4 | **REINFORCE on the LLM** | Your stage 1 loss on real completions, plus a KL penalty to a frozen copy of the starting model; watching for reward hacking | 4090, ~15 min |
| 5 | **GRPO** | Groups of completions per prompt, group-relative advantages, then reusing a batch for several updates with probability ratios and **clipping** (the PPO idea, without a critic) | 4090, ~20 min |
| 6 | **PPO** | A learned value function in place of the group mean, per-token advantages with GAE; what the critic buys and what it costs | 4090, ~30 min |
| 7 | **DPO** | Learning from preference pairs offline, with no sampling loop. Derived from the same KL-regularized objective, trained on pairs built from your own stage 3 samples | 4090, ~10 min |
| 8 | **Your code vs. TRL** (optional) | Re-run GRPO and DPO with Hugging Face TRL and map every config option to a line you wrote | 4090 |

GRPO comes before PPO on purpose: it's REINFORCE plus a group baseline plus clipping, so each step is small. PPO then adds the one remaining piece, the critic.

## How each stage works

1. Read the notebook top to bottom. Fill in the `# YOUR CODE HERE` cells; the check cell under each tells you if it's right.
2. Answer the **Think** questions in your own words *before* running the next cell.
3. Do the experiments at the end and write down what you saw.
4. Post questions and surprises in the project thread. We go through them, then build the next stage together.

Reference solutions (including answers to the Think questions) are in `solutions/`. Look only when stuck.

## Course authoring

The exercise and solution notebooks are both generated from `tools/lessons/*.py` by `tools/build_notebooks.py`. It never overwrites an exercise notebook that already exists, so your work in `notebooks/` is safe.
