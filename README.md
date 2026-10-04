# Learning RL, from policy gradients to LLM post-training

A hands-on course for an ML engineer who already knows deep learning. Every lesson
contains complete code, an explanatory notebook, experiments, and interpretation
questions. Start with [the study plan](STUDY_PLAN.md).

**Available now:** lesson 1, contextual bandits and REINFORCE. Lessons 2–6 are planned,
not implemented yet. We will build them incrementally after discussing each lesson.

## Run the first experiment

From this folder:

```bash
python3.12 -m venv .venv  # Already created in this workspace.
source .venv/bin/activate
python scripts/preflight.py
python -m unittest discover -s tests -v
python scripts/run_bandit.py
```

No packages or GPU are needed for the default backend. The policy is a table of
three logits per context; its exact softmax gradient is written explicitly.
This isolates REINFORCE before adding neural-network/autograd machinery.

The experiment runs three baseline variants across five seeds with equal sample
budgets. A new timestamped directory under `runs/` contains `config.json`,
`metrics.csv`, `frozen_variance.json`, `learning_curves.svg`, and `summary.md`.
Open the summary or SVG locally or copy them to your Mac using `scp`. Output
directories are never overwritten. Exact installed package versions are recorded
in each run's configuration.

For a quick check or a longer experiment:

```bash
python scripts/run_bandit.py --steps 100 --seeds 0
python scripts/run_bandit.py --steps 2000 --batch-size 64 --seeds 0 1 2 3 4
```

## Read and run the notebook

The lesson is [notebooks/01_reinforce_bandit.ipynb](notebooks/01_reinforce_bandit.ipynb).

On a shell with package-index access:

```bash
source .venv/bin/activate
python -m pip install -r requirements/notebooks.txt
python -m jupyterlab --no-browser --ip=127.0.0.1 --port=8888
```

On your Mac, forward the port (replace `YOUR_PC` with your SSH host):

```bash
ssh -N -L 8888:127.0.0.1:8888 YOUR_PC
```

Open the token-bearing localhost URL printed by Jupyter. Keep authentication enabled.
The CLI and notebook import the same training implementation, so the explanations
and actual experiments stay in sync.

For headless verification with a real Jupyter kernel after installing those packages:

```bash
mkdir -p runs/notebook-check
python -m jupyter nbconvert --to notebook --execute notebooks/01_reinforce_bandit.ipynb --ExecutePreprocessor.timeout=300 --output 01_reinforce_bandit.executed --output-dir runs/notebook-check
```

Without Jupyter installed, the test suite executes every Python cell directly.
That checks the computations, but not kernel startup or notebook rendering.

## Optional PyTorch backend

Install a suitable wheel using the [official PyTorch instructions](https://pytorch.org/get-started/locally/),
or use these requirements for the CPU lesson:

```bash
python -m pip install -r requirements/torch.txt
python -m unittest discover -s tests -v
python scripts/run_bandit.py --backend torch
```

Read `src/rl_study/torch_bandit.py` beside the explicit gradient in `bandit.py`.
The policy loss is `-(advantage.detach() * log_probability).mean()`; the value
loss has a separate backward pass. Both backends use SGD and identical sampled
rollouts, and a test compares their updates numerically when PyTorch is available.

## GitHub setup

Target remote: `git@github.com:amirarsalan90/learning_RL.git`.
This managed session exposes `.git` as an empty read-only placeholder, so Git
initialization here is blocked. From your normal unrestricted SSH shell, if the
folder is still not a Git repository:

```bash
cd /home/arsalan/Desktop/learning_RL
git init -b main
git remote add origin git@github.com:amirarsalan90/learning_RL.git
git add .gitignore README.md STUDY_PLAN.md requirements src scripts tests notebooks
git commit -m "Add RL study roadmap and contextual REINFORCE lesson"
git push -u origin main
```

If it is already initialized, inspect `git status` and `git remote -v` first.
Do not delete or replace `.git`. Generated artifacts and the virtual environment
are ignored; source, lessons, and requirements are tracked.

## Current environment limits

Python 3.12 and the local `.venv` are available. This session could not install
packages from its configured index, resolve GitHub, or access an NVIDIA device.
The dependency-free experiment remains runnable. Before LLM training, install a
CUDA-enabled PyTorch build and require a successful preflight:

```bash
python scripts/preflight.py --require-cuda
```

That distinguishes actual PyTorch CUDA availability from simply having a GPU in
the PC. Driver/system changes are outside this repository's setup.
