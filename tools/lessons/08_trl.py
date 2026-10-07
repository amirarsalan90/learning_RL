"""Notebook 8: the same GRPO and DPO runs with Hugging Face TRL."""


def md(t):
    return ("md", t)


def code(t):
    return ("code", t)


CELLS = [
    md(r"""
# 8 · The same thing with TRL

**Run-only. Needs the GPU; about 30–60 minutes for both runs.**

[TRL](https://github.com/huggingface/trl) is Hugging Face's library for RL on language models, and what most people actually use. It implements the same algorithms as `rlcourse/`, with many more options and far more engineering (multi-GPU, vLLM generation, memory tricks). This notebook runs GRPO and DPO with TRL on the same task, with settings chosen to match notebooks 4 and 6, and maps each setting to the line of our code it corresponds to.

Install it first (once): `uv sync --group trl`
"""),
    code(r"""
import json
import time
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, DPOConfig, DPOTrainer
from rlcourse import llm, task, viz, grpo, dpo

tok = llm.load_tokenizer()
cuda = torch.cuda.is_available()
SMOKE = llm.SMOKE
"""),
    md(r"""
## GRPO: our config → TRL's

| our `grpo.Config` | TRL `GRPOConfig` | meaning |
|---|---|---|
| `samples_per_prompt = 4` | `num_generations = 4` | group size |
| `prompts_per_step = 16` | `per_device_train_batch_size × gradient_accumulation_steps = 64` completions | completions per batch |
| `updates_per_batch = 2` | `num_iterations = 2` | gradient steps per generated batch |
| `steps = 80` | `max_steps = 160` | TRL counts *gradient* steps: 80 batches × 2 |
| `clip_eps = 0.2` | `epsilon = 0.2` | the PPO clip |
| `kl_coef = 0.02` | `beta = 0.02` | KL leash to the reference (TRL's default is 0: no reference model at all) |
| `advantages()`: ÷ group std | `scale_rewards = "group"` | GRPO's normalization |
| loss ÷ total tokens in the batch | `loss_type = "dapo"` | how token losses are averaged |
| `max_new_tokens = 200` | `max_completion_length = 200` | |
| `temperature = 1.0`, no top-k/top-p | `temperature = 1.0`, `top_k = 0`, `top_p = 1.0` | sample from the true policy |
| `lr = 2e-6`, constant | `learning_rate = 2e-6`, `lr_scheduler_type = "constant"` | |
| `task.reward` | `reward_funcs = [correctness]` | the reward function |

The dataset is just prompts (as chat messages) plus the column the reward function needs (`answer`). TRL passes every extra dataset column to the reward function as a keyword argument.
"""),
    code(r"""
gcfg = llm.maybe_smoke(grpo.Config())
problems = task.TrainStream(gcfg.difficulty, gcfg.seed).next(gcfg.steps * gcfg.prompts_per_step)
train_ds = Dataset.from_list([
    {"prompt": [{"role": "system", "content": task.SYSTEM_PROMPT}, {"role": "user", "content": p.question}],
     "answer": p.answer}
    for p in problems
])


def correctness(completions, answer, **kwargs):
    # completions arrive as chat messages: [{"role": "assistant", "content": "..."}]
    return [task.reward(c[0]["content"], a) for c, a in zip(completions, answer)]


completions_per_step = gcfg.prompts_per_step * gcfg.samples_per_prompt
args = GRPOConfig(
    output_dir=str(llm.RUNS / "trl-grpo"),
    num_generations=gcfg.samples_per_prompt,
    per_device_train_batch_size=gcfg.micro_batch,
    gradient_accumulation_steps=completions_per_step // gcfg.micro_batch,
    num_iterations=gcfg.updates_per_batch,
    max_steps=gcfg.steps * gcfg.updates_per_batch,
    epsilon=gcfg.clip_eps,
    beta=gcfg.kl_coef,
    scale_rewards="group",
    loss_type="dapo",
    max_completion_length=gcfg.max_new_tokens,
    temperature=gcfg.temperature, top_k=0, top_p=1.0,
    learning_rate=gcfg.lr, lr_scheduler_type="constant", max_grad_norm=gcfg.max_grad_norm,
    logging_steps=1, report_to="none", save_strategy="no",
    bf16=cuda, use_cpu=not cuda, gradient_checkpointing=False,
)
print({k: getattr(args, k) for k in ["num_generations", "per_device_train_batch_size", "gradient_accumulation_steps",
                                      "num_iterations", "max_steps", "beta", "epsilon", "loss_type"]})
"""),
    code(r"""
trainer = GRPOTrainer(model=llm.MODEL_NAME, reward_funcs=correctness, args=args,
                      train_dataset=train_ds, processing_class=tok)
torch.cuda.reset_peak_memory_stats() if cuda else None
t0 = time.time()
trainer.train()
minutes = (time.time() - t0) / 60
print(f"{minutes:.1f} minutes, peak GPU memory {llm.peak_memory_gb():.1f} GB")
"""),
    md(r"""
TRL logs one row per gradient step; rows with a `reward` are the first update on each new batch. Plotted next to our own GRPO run from notebook 4 (if you ran it):
"""),
    code(r"""
logs = [h for h in trainer.state.log_history if "reward" in h]
trl_history = [{"reward": h["reward"], "kl": h.get("kl", 0.0), "completion_tokens": h["completions/mean_length"],
                "zero_variance_groups": h.get("frac_reward_zero_std", 0.0),
                "generated_tokens": h["completions/mean_length"] * completions_per_step} for h in logs]
runs = {"TRL GRPO": trl_history}
ours = llm.load_run("grpo")
if ours:
    runs = {"our GRPO": ours["history"], "TRL GRPO": trl_history}
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for ax, key, title in zip(axes, ["reward", "kl", "zero_variance_groups"], ["reward", "KL to reference", "zero-variance groups"]):
    for (name, hist), c in zip(runs.items(), [viz.PALETTE["green"], viz.PALETTE["blue"]]):
        ys = [h[key] for h in hist]
        ax.plot(ys, color=c, alpha=0.3), ax.plot(viz.smooth(ys), color=c, lw=2, label=name)
    ax.set_title(title), ax.set_xlabel("batch")
axes[0].legend(), fig.tight_layout(), plt.show()
"""),
    code(r"""
results = llm.evaluate(trainer.model, tok)
results.update(minutes=minutes, peak_memory_gb=llm.peak_memory_gb())
llm.save_run("trl-grpo", gcfg, trl_history, results, [])
print(f"TRL GRPO held-out accuracy: {results['accuracy']:.1%}"
      + (f"   (ours: {ours['results']['accuracy']:.1%})" if ours else ""))
del trainer
torch.cuda.empty_cache() if cuda else None
"""),
    md(r"""
## DPO: our config → TRL's

| our `dpo.Config` | TRL `DPOConfig` |
|---|---|
| `beta = 0.1` | `beta = 0.1` |
| `lr = 1e-6` | `learning_rate = 1e-6` |
| `pairs_per_step = 16` | `per_device_train_batch_size × gradient_accumulation_steps = 16` |
| `epochs = 2` | `num_train_epochs = 2` |
| `dpo_loss()` | `loss_type = "sigmoid"` (the default, the original DPO loss) |
| `reference_logprobs()` up front | the reference model is created for you; `precompute_ref_log_probs = True` does it up front |

The pairs are the ones notebook 6 built and saved (same prompts, same replies). If you skipped notebook 6, they're rebuilt here.
"""),
    code(r"""
dcfg = llm.maybe_smoke(dpo.Config())
pairs_path = llm.RUNS / "dpo" / "pairs.json"
if pairs_path.exists() and not SMOKE:
    pairs = json.loads(pairs_path.read_text())
else:
    ref = llm.load_model(trainable=False)
    built, _ = dpo.build_pairs(ref, tok, dcfg)
    pairs = [{"question": p["question"], "chosen": p["chosen"]["text"], "rejected": p["rejected"]["text"]} for p in built]
    del ref
print(len(pairs), "pairs")


def as_messages(p):
    prompt = [{"role": "system", "content": task.SYSTEM_PROMPT}, {"role": "user", "content": p["question"]}]
    return {"prompt": prompt, "chosen": [{"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "assistant", "content": p["rejected"]}]}


dpo_ds = Dataset.from_list([as_messages(p) for p in pairs])
dargs = DPOConfig(
    output_dir=str(llm.RUNS / "trl-dpo"), beta=dcfg.beta, learning_rate=dcfg.lr, lr_scheduler_type="constant",
    per_device_train_batch_size=dcfg.micro_batch, gradient_accumulation_steps=max(1, dcfg.pairs_per_step // dcfg.micro_batch),
    num_train_epochs=dcfg.epochs, max_grad_norm=dcfg.max_grad_norm, max_length=1024,
    logging_steps=1, report_to="none", save_strategy="no", bf16=cuda, use_cpu=not cuda, gradient_checkpointing=False,
)
dtrainer = DPOTrainer(model=llm.MODEL_NAME, args=dargs, train_dataset=dpo_ds, processing_class=tok)
t0 = time.time()
dtrainer.train()
dminutes = (time.time() - t0) / 60
"""),
    code(r"""
dlogs = [h for h in dtrainer.state.log_history if "rewards/chosen" in h]
ours = llm.load_run("dpo")
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, (title, hist, keys) in zip(axes, [
        ("TRL DPO", dlogs, ("rewards/chosen", "rewards/rejected")),
        ("our DPO (notebook 6)", ours["history"] if ours else [], ("chosen_reward", "rejected_reward"))]):
    for key, c, label in zip(keys, [viz.PALETTE["green"], viz.PALETTE["red"]], ["chosen", "rejected"]):
        ys = [h[key] for h in hist]
        if ys:
            ax.plot(viz.smooth(ys), color=c, lw=2, label=label)
    ax.axhline(0, color="black", lw=0.6), ax.set_title(f"{title}: implicit rewards"), ax.set_xlabel("step"), ax.legend()
fig.tight_layout(), plt.show()

results = llm.evaluate(dtrainer.model, tok)
results.update(minutes=dminutes, peak_memory_gb=llm.peak_memory_gb())
llm.save_run("trl-dpo", dcfg, [], results, [])
print(f"TRL DPO held-out accuracy: {results['accuracy']:.1%}" + (f"   (ours: {ours['results']['accuracy']:.1%})" if ours else ""))
"""),
    md(r"""
## What to take away

- Every TRL knob you just set corresponds to a line you've read in `rlcourse/`. When a TRL run behaves strangely, that mapping is how you reason about it.
- Defaults matter and differ: TRL's GRPO uses `beta = 0` (no reference model, no KL) and the `"dapo"` token averaging. Different papers and libraries make different choices for these, and they change results.
- Small differences between our runs and TRL's are expected (random seeds, numerics, details like how truncated completions are handled). Large ones mean a setting doesn't match.

**Where to go from here**: try `difficulty="hard"`, a bigger model (`Qwen/Qwen2.5-1.5B-Instruct` fits on a 24 GB GPU with LoRA via `peft`), or a different reward, like a formatting bonus, and watch for reward hacking.
"""),
]

CELLS.append(md(r"""
## Further reading

- [TRL documentation](https://huggingface.co/docs/trl): GRPOTrainer and DPOTrainer, with every config option explained.
"""))
