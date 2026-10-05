# RL for LLMs, walked through

REINFORCE → GRPO → PPO → DPO, each implemented in ~100 readable lines of PyTorch and walked through in a notebook you just **run and read**. You don't write any code. Every notebook has diagrams, live training plots, token-level heatmaps of what the model learned, and side-by-side code diffs showing exactly what each algorithm changes from the previous one. They all train the same small model (`Qwen2.5-0.5B-Instruct`) on the same arithmetic task on a single RTX 4090, so the results compare directly.

| | notebook | GPU time |
|---|---|---|
| 1 | [The one idea behind all of them: policy gradients](notebooks/01_policy_gradient_bandit.ipynb) (a bandit; already run, outputs included) | none |
| 2 | [An LLM is a policy](notebooks/02_llm_as_policy.ipynb): sampling, token log-probs, reward, starting accuracy | ~5 min |
| 3 | [REINFORCE](notebooks/03_reinforce.ipynb) | ~15–30 min |
| 4 | [GRPO](notebooks/04_grpo.ipynb): group baselines and clipped updates | ~20–40 min |
| 5 | [PPO](notebooks/05_ppo.ipynb): a critic that scores every token | ~30–50 min |
| 6 | [DPO](notebooks/06_dpo.ipynb): offline preference pairs | ~10–20 min |
| 7 | [Side by side](notebooks/07_comparison.ipynb): all results, one table, a cheat sheet | none |
| 8 | [The same thing with TRL](notebooks/08_trl.ipynb), Hugging Face's library | ~30–60 min |

GPU times are estimates; the first real runs will tell.

## Setup

See **[docs/setup.md](docs/setup.md)** for working from a Mac with everything running on a GPU PC over Tailscale. In short, on the PC:

```bash
uv sync --group trl
```

then open a notebook in VS Code / Cursor (Remote-SSH) with the `.venv` kernel and run all cells.

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
tools/           regenerates the notebooks and diagrams; smoke tests
```

## For course maintenance

Notebooks are generated from `tools/lessons/*.py` by `tools/build_notebooks.py`; diagrams by `tools/make_figures.py`. To check every notebook runs end to end on a CPU with a tiny random model:

```bash
uv sync --all-groups
uv run python tools/make_tiny_model.py /tmp/tiny-qwen
RLCOURSE_MODEL=/tmp/tiny-qwen RLCOURSE_SMOKE=1 uv run python tools/build_notebooks.py --check
```
