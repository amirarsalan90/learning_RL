# Setup: code on the Mac, run on the 4090

**Recommendation:** use VS Code (or Cursor) with the **Remote - SSH** extension to connect from the Mac to the PC over Tailscale. The editor window is on your Mac; the files, the Python environment, the notebook kernel and the GPU are all on the PC. You open `.ipynb` files in the editor as usual, and every cell runs on the 4090.

Why this over a browser Jupyter server: one tool for notebooks, `.py` files, a terminal and git; no tokens or port forwards to juggle; and the editor reconnects on its own when the Mac sleeps.

Stage 1 doesn't need a GPU at all, so you can start it on the Mac (step 5 works locally too) while you set up the PC.

## 1. The PC: Linux with the NVIDIA driver

**If the PC runs Linux:** install the NVIDIA driver from your distro's packages and check that `nvidia-smi` lists the RTX 4090. That's it; skip to step 2.

**If it runs Windows:** use WSL2 (Ubuntu). It's a real Linux userland with GPU access, and every RL/LLM tool assumes Linux.

1. Install the latest NVIDIA *Windows* driver (the normal Game Ready or Studio driver). Do **not** install a Linux driver inside WSL.
2. In an admin PowerShell: `wsl --install -d Ubuntu-24.04`, reboot, and create your Linux user.
3. In Ubuntu, enable systemd so services (SSH, Tailscale) start on their own:
   ```bash
   printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
   ```
   then `wsl --shutdown` in PowerShell and reopen Ubuntu.
4. Check: `nvidia-smi` inside Ubuntu lists the 4090.

## 2. Make the PC reachable over Tailscale with SSH

On the Linux side (inside WSL if Windows):

```bash
# Tailscale. On Windows this makes WSL its own device on your tailnet, which is
# what lets the Mac SSH straight into Linux without any port forwarding.
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname rtx-pc

# SSH server
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
```

On the Mac:

```bash
ssh-keygen -t ed25519            # skip if ~/.ssh/id_ed25519 already exists
ssh-copy-id YOUR_LINUX_USER@rtx-pc
```

Add this to `~/.ssh/config` on the Mac:

```
Host rtx
    HostName rtx-pc
    User YOUR_LINUX_USER
    ServerAliveInterval 30
```

Check: `ssh rtx nvidia-smi` from the Mac prints the 4090's status.

`rtx-pc` resolves through Tailscale's MagicDNS (on by default). If it doesn't, use the PC's `100.x.y.z` address from `tailscale ip -4`.

## 3. The project on the PC

```bash
ssh rtx
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv manages Python and packages
git clone https://github.com/amirarsalan90/learning_RL.git
cd learning_RL
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line should print `True NVIDIA GeForce RTX 4090`. On Linux, the default PyTorch wheels already include CUDA, so nothing else is needed. If it prints `False`, the NVIDIA driver is probably too old for the CUDA version PyTorch was built with: update the driver (on Windows, the Windows driver) and try again.

## 4. Connect the editor

1. In VS Code on the Mac, install the **Remote - SSH**, **Python** and **Jupyter** extensions. (Cursor: the same, from its extension panel.)
2. `Cmd+Shift+P` → **Remote-SSH: Connect to Host…** → `rtx`.
3. **File → Open Folder** → `~/learning_RL`.
4. In the remote window, install the **Python** and **Jupyter** extensions again when it offers (they need to exist on the remote side).
5. Open `notebooks/01_policy_gradient_bandit.ipynb`, click **Select Kernel** (top right) → **Python Environments** → `.venv`.

Now every cell runs on the PC.

## 5. Running locally on the Mac instead (stage 1 only)

```bash
git clone https://github.com/amirarsalan90/learning_RL.git && cd learning_RL
uv sync
code .   # then open the notebook and pick the .venv kernel
```

## Long runs

From stage 4 onward some training runs take tens of minutes. Run those as scripts inside `tmux` on the PC, so they keep going if the Mac sleeps or the connection drops:

```bash
ssh rtx
tmux new -s train        # later: tmux attach -t train
uv run python ...        # detach with Ctrl-b then d
```

Also set the PC to never sleep (Windows: Settings → System → Power → Sleep: Never).

## Alternative: a plain Jupyter Lab server

If you'd rather use Jupyter in the browser:

```bash
# on the PC, inside tmux
cd ~/learning_RL
uv sync --group jupyter
uv run jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

```bash
# on the Mac
ssh -N -L 8888:127.0.0.1:8888 rtx
```

Then open the `http://127.0.0.1:8888/lab?token=…` URL that Jupyter printed. Keeping Jupyter bound to `127.0.0.1` and tunnelling over SSH means it's never exposed, even on your tailnet.
