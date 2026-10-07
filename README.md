# RL for LLMs, walked through

If you've read about RLHF, PPO, DPO or GRPO and came away with the pieces jumbled together, this is for you. It's a set of notebooks you **run and read**, not exercises: you don't write any code.

REINFORCE → GRPO → PPO → DPO are each implemented in ~100 readable lines of PyTorch. Every notebook walks through one of them with diagrams, live training plots, token-level heatmaps of what the model learned, and side-by-side code diffs showing exactly what each algorithm changes from the previous one. They all train the same small model (`Qwen2.5-0.5B-Instruct`) on the same arithmetic task, so the results compare directly. The last notebook does the same with Hugging Face [TRL](https://github.com/huggingface/trl) and maps each of its settings back to our code.

| | notebook | GPU time |
|---|---|---|
| 1 | [The one idea behind all of them: policy gradients](notebooks/01_policy_gradient_bandit.ipynb) (a bandit; outputs included, runs on CPU) | none |
| 2 | [An LLM is a policy](notebooks/02_llm_as_policy.ipynb): sampling, token log-probs, reward, starting accuracy | ~5 min |
| 3 | [REINFORCE](notebooks/03_reinforce.ipynb) | ~15–30 min |
| 4 | [GRPO](notebooks/04_grpo.ipynb): group baselines and clipped updates | ~20–40 min |
| 5 | [PPO](notebooks/05_ppo.ipynb): a critic that scores every token | ~30–50 min |
| 6 | [DPO](notebooks/06_dpo.ipynb): offline preference pairs | ~10–20 min |
| 7 | [Side by side](notebooks/07_comparison.ipynb): all results, one table, a cheat sheet | none |
| 8 | [The same thing with TRL](notebooks/08_trl.ipynb) | ~30–60 min |

GPU times are rough estimates for one 24 GB card such as an RTX 4090.

## What you need

- **To read along:** nothing. The notebooks render on GitHub.
- **To run notebook 1:** any computer.
- **To run notebooks 2–8:** Linux (or Windows with WSL2) and an NVIDIA GPU with about 24 GB of memory. Smaller cards work with the memory settings in [docs/setup.md](docs/setup.md#less-gpu-memory).

## Quick start

```bash
git clone https://github.com/amirarsalan90/learning_RL.git
cd learning_RL
uv sync --group trl
```

Then open `notebooks/01_policy_gradient_bandit.ipynb` in VS Code or Cursor, pick the `.venv` kernel, and run all cells. Prefer Jupyter Lab, or running on a remote GPU machine? See **[docs/setup.md](docs/setup.md)**.

## Where the code lives

```
rlcourse/
  task.py        the arithmetic problems and the 0/1 reward
  llm.py         shared plumbing: load, sample, token log-probs, evaluate
  reinforce.py   REINFORCE + running baseline + KL penalty
  grpo.py        reinforce.py + group advantages + clipped reuse
  ppo.py         grpo.py's update + a critic, GAE, KL-in-reward
  dpo.py         offline preference loss
  viz.py         plots, token heatmaps, code diffs
notebooks/       the walkthroughs (+ figures/)
runs/            results each notebook saves (created when you run them)
tools/           generates the notebooks and diagrams
```

## Further reading

The papers behind each notebook, in the order they come up:

- R. J. Williams (1992), [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696): REINFORCE.
- R. Sutton and A. Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), chapter 13: policy gradients.
- A. Ahmadian et al. (2024), [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740).
- Z. Shao et al. (2024), [DeepSeekMath](https://arxiv.org/abs/2402.03300): introduces GRPO.
- DeepSeek-AI (2025), [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- J. Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- J. Schulman et al. (2015), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- L. Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT).
- R. Rafailov et al. (2023), [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

Each notebook ends with its own short list.

## License

[MIT](LICENSE)
