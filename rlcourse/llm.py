"""The plumbing every LLM algorithm shares: load the model, sample completions, score
them, and compute per-token log-probabilities.

Shapes, used everywhere in this course:
    input_ids, attention_mask, completion_mask, logprobs : (batch, seq_len)
A row is [left padding | prompt | completion | right padding]. completion_mask is 1
on completion tokens (including the final end-of-turn token) and 0 elsewhere.
logprobs[:, t] = log π(token t | tokens before t), with logprobs[:, 0] = 0.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from . import task

MODEL_NAME = os.environ.get("RLCOURSE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
SMOKE = os.environ.get("RLCOURSE_SMOKE") == "1"  # tiny CPU runs that only check the code paths
ROOT = Path(__file__).resolve().parent.parent

if SMOKE:  # a random tiny model never answers correctly; random rewards keep every code path busy
    import random as _random
    task.reward = lambda text, answer: float(_random.random() < 0.4)
RUNS = ROOT / "runs"


def maybe_smoke(cfg):
    """Shrink a config for the automated CPU smoke test. On your GPU this returns cfg unchanged."""
    if not SMOKE:
        return cfg
    small = {"steps": 3, "prompts_per_step": 8, "max_new_tokens": 24, "n_prompts": 32, "epochs": 1,
             "pairs_per_step": 4, "micro_batch": 4}
    for k, v in small.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


def load_tokenizer(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name=MODEL_NAME, trainable=True):
    """The policy is kept in float32 (so tiny learning-rate updates aren't rounded away)
    and run in bfloat16 through autocast. Frozen copies (the reference model) are bf16."""
    dtype = torch.float32 if trainable else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEVICE)
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or model.config.eos_token_id
    if not trainable:
        model.eval().requires_grad_(False)
    return model


def autocast():
    return torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda")


def chat_prompt(tok, question):
    messages = [{"role": "system", "content": task.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class Rollout:
    """A batch of sampled completions and everything we know about them."""

    problems: list
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    texts: list
    rewards: torch.Tensor
    group: torch.Tensor  # which prompt each row came from (rows of one group share a prompt)
    gen_seconds: float = 0.0
    extras: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.texts)

    def subset(self, idx):
        idx = torch.as_tensor(idx)
        return Rollout(
            [self.problems[i] for i in idx.tolist()], self.input_ids[idx], self.attention_mask[idx],
            self.completion_mask[idx], [self.texts[i] for i in idx.tolist()], self.rewards[idx],
            self.group[idx], self.gen_seconds, {k: v[idx] for k, v in self.extras.items()},
        )


def end_token_ids(tok):
    ids = {tok.eos_token_id, tok.pad_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tok.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


@torch.no_grad()
def generate(model, tok, problems, samples_per_prompt=1, max_new_tokens=200, temperature=1.0, greedy=False):
    """Sample `samples_per_prompt` completions for each problem and score them.

    Sampling is plain ancestral sampling from the softmax at `temperature`: no top-k,
    top-p or repetition penalty. Policy-gradient math assumes the completions were drawn
    from exactly the distribution whose log-probs we later compute.
    """
    t0 = time.time()
    model.eval()
    prompts = [chat_prompt(tok, p.question) for p in problems for _ in range(samples_per_prompt)]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    ends = end_token_ids(tok)
    cfg = GenerationConfig(
        do_sample=not greedy, temperature=temperature if not greedy else None, top_k=0 if not greedy else None,
        top_p=1.0 if not greedy else None, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
        eos_token_id=ends, pad_token_id=tok.pad_token_id,
    )
    with autocast():
        out = model.generate(**enc, generation_config=cfg)
    prompt_len = enc.input_ids.shape[1]
    completion = out[:, prompt_len:]
    is_end = torch.isin(completion, torch.tensor(ends, device=out.device))
    # keep tokens up to and including the first end token
    keep = (is_end.cumsum(1) - is_end.long()) == 0
    completion_mask = torch.cat([torch.zeros_like(enc.attention_mask), keep.long()], 1)
    attention_mask = torch.cat([enc.attention_mask, keep.long()], 1)
    texts = [tok.decode(c[k], skip_special_tokens=True) for c, k in zip(completion, keep)]
    expanded = [p for p in problems for _ in range(samples_per_prompt)]
    rewards = torch.tensor([task.reward(t, p.answer) for t, p in zip(texts, expanded)])
    group = torch.arange(len(problems)).repeat_interleave(samples_per_prompt)
    return Rollout(expanded, out, attention_mask, completion_mask, texts, rewards, group, time.time() - t0)


def position_ids(attention_mask):
    # With left padding, positions must start at 0 on the first real token, as in generate().
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def token_logprobs(model, input_ids, attention_mask):
    """log π(token t | tokens < t) for every position; gradients flow if model is trainable."""
    with autocast():
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids(attention_mask)).logits
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    lp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return torch.cat([torch.zeros_like(lp[:, :1]), lp], dim=1)


@torch.no_grad()
def batched_logprobs(model, rollout, micro_batch=8):
    """token_logprobs over a whole rollout without gradients, in micro-batches to save memory."""
    out = []
    for i in range(0, len(rollout), micro_batch):
        sl = slice(i, i + micro_batch)
        out.append(token_logprobs(model, rollout.input_ids[sl], rollout.attention_mask[sl]))
    return torch.cat(out)


def micro_batches(n, size):
    for i in range(0, n, size):
        yield list(range(i, min(i + size, n)))


@torch.no_grad()
def evaluate(model, tok, problems=None, batch_size=64, max_new_tokens=200):
    """Greedy-decoding accuracy on the held-out problems."""
    problems = problems or task.eval_problems(n=16 if SMOKE else 200)
    texts, rewards = [], []
    for i in range(0, len(problems), batch_size):
        r = generate(model, tok, problems[i:i + batch_size], greedy=True, max_new_tokens=max_new_tokens)
        texts += r.texts
        rewards += r.rewards.tolist()
    formatted = [task.parse_answer(t) is not None for t in texts]
    return {
        "accuracy": sum(rewards) / len(rewards),
        "format_rate": sum(formatted) / len(formatted),
        "mean_chars": sum(len(t) for t in texts) / len(texts),
        "examples": [{"question": p.question, "answer": p.answer, "completion": t, "reward": r}
                     for p, t, r in list(zip(problems, texts, rewards))[:20]],
    }


def rollout_stats(r):
    """Per-step numbers logged by every algorithm."""
    lengths = r.completion_mask.sum(1).float()
    return {
        "reward": r.rewards.mean().item(),
        "format_rate": sum(task.parse_answer(t) is not None for t in r.texts) / len(r),
        "completion_tokens": lengths.mean().item(),
        "generated_tokens": int(lengths.sum().item()),
        "gen_seconds": r.gen_seconds,
    }


def save_run(name, cfg, history, results, samples):
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(cfg) if cfg is not None else {}, "history": history, "results": results, "samples": samples}
    (out / "run.json").write_text(json.dumps(payload, indent=1, default=str))
    return out


def load_run(name):
    path = RUNS / name / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def peak_memory_gb():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
