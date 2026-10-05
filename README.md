# Learning RL for LLMs, by implementing it

A hands-on course: REINFORCE → GRPO → PPO → DPO, each written from scratch in plain PyTorch, in notebooks where **you** fill in the key code. It starts from a 5-armed bandit and ends with RL fine-tuning a small LLM on a single RTX 4090.

- **[ROADMAP.md](ROADMAP.md)** — the stages and what each one teaches.
- **[docs/setup.md](docs/setup.md)** — working from a Mac with the code running on a GPU PC over Tailscale.

## Quick start

```bash
uv sync
```

Then open `notebooks/01_policy_gradient_bandit.ipynb` in VS Code / Cursor (or Jupyter) with the `.venv` kernel and start at the top.

## Layout

```
notebooks/   exercise notebooks, the ones you work in
solutions/   the same notebooks, filled in, with answers to the Think questions
rlcourse/    small shared helpers (the checks that grade your exercise cells)
tools/       the generator for notebooks/ and solutions/
docs/        setup guide
```
