# Setup

## What you need

- **To read:** nothing. Open the notebooks on GitHub; any notebook saved with its outputs shows every plot and sample.
- **To run notebook 1:** any computer with Python. It runs on a CPU in under a minute.
- **To run notebooks 2–8:** Linux (or Windows with WSL2) and an NVIDIA GPU with about **24 GB** of memory, such as an RTX 3090, 4090 or A5000. The model is small (0.5B parameters), but every notebook fully fine-tunes it and keeps a frozen copy, and PPO adds a second trainable model.

## Install

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and packages. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/amirarsalan90/learning_RL.git
cd learning_RL
uv sync --group trl     # --group trl adds Hugging Face TRL, used only in notebook 8
```

Check that PyTorch sees the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

It should print `True` and your GPU's name. If it prints `False`, the NVIDIA driver is usually too old for the CUDA version PyTorch was built with: update the driver and try again. On Linux, the default PyTorch wheels already include CUDA.

The first LLM notebook downloads `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB) from Hugging Face automatically. No account is needed.

## Open the notebooks

Any Jupyter front end works, as long as it uses the project's `.venv` environment:

- **VS Code or Cursor:** open the folder, open a notebook, click **Select Kernel** → **Python Environments** → `.venv`. This also works on a remote GPU machine through the Remote - SSH extension.
- **Jupyter Lab in the browser:**
  ```bash
  uv sync --group trl --group jupyter   # uv keeps only the groups you list
  uv run jupyter lab
  ```

Then run the cells from top to bottom.

## Windows

Use WSL2: install the normal NVIDIA Windows driver (not a Linux driver inside WSL), run `wsl --install -d Ubuntu-24.04`, and follow the Linux steps inside Ubuntu. `nvidia-smi` inside Ubuntu should list your GPU.

## Long runs

Notebooks 3–6 and 8 train for roughly 15–50 minutes each. If you work on a remote machine and your laptop might sleep or disconnect, run Jupyter Lab inside `tmux` on the GPU machine so the kernel keeps going with nobody connected:

```bash
tmux new -s jupyter          # later: tmux attach -t jupyter; detach with Ctrl-b then d
uv run jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

and reach it with an SSH tunnel: `ssh -N -L 8888:127.0.0.1:8888 your-gpu-machine`, then open the URL Jupyter printed.

Every notebook also saves its results to `runs/`, so a finished run's numbers survive even if the live plot was lost.

## Less GPU memory

Each notebook's config is printed before training. Lowering `micro_batch` (completions per forward pass) reduces peak memory without changing the algorithm; lowering `max_new_tokens` or `prompts_per_step` also helps, but changes the experiment. These settings have only been sized for 24 GB.
