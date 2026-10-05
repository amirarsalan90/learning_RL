"""DPO (Direct Preference Optimization) for an LLM.

DPO is *offline*: there's no sampling or reward function inside the training loop.
Instead you start from a fixed dataset of pairs (prompt, chosen reply, rejected reply)
and push the model to prefer the chosen one, while a frozen reference model keeps it
from drifting.

Here we build the pairs ourselves: sample several replies per prompt from the starting
model once, and pair a correct reply (chosen) with an incorrect one (rejected).

The loss, per pair:
    margin = beta * [(log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))]
    loss   = -log sigmoid(margin)
It comes from the same KL-regularized objective as PPO, solved in closed form, which
is why no reward model or critic is needed.
"""

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import llm, task


@dataclass
class Config:
    n_prompts: int = 512
    samples_per_prompt: int = 4
    max_new_tokens: int = 200
    temperature: float = 1.0
    epochs: int = 2
    pairs_per_step: int = 16
    lr: float = 1e-6
    beta: float = 0.1
    micro_batch: int = 8
    max_grad_norm: float = 1.0
    difficulty: str = "medium"
    seed: int = 0


def build_pairs(model, tok, cfg, batch_prompts=32):
    """Sample from the starting model once; keep one (correct, incorrect) pair per prompt."""
    stream = task.TrainStream(cfg.difficulty, cfg.seed)
    pairs, n_all_right, n_all_wrong, generated = [], 0, 0, 0
    for _ in range(0, cfg.n_prompts, batch_prompts):
        r = llm.generate(model, tok, stream.next(batch_prompts), cfg.samples_per_prompt,
                         cfg.max_new_tokens, cfg.temperature)
        generated += int(r.completion_mask.sum())
        for g in range(batch_prompts):
            rows = (r.group == g).nonzero().squeeze(-1).tolist()
            good = [i for i in rows if r.rewards[i] == 1]
            bad = [i for i in rows if r.rewards[i] == 0]
            if not bad:
                n_all_right += 1
            elif not good:
                n_all_wrong += 1
            else:
                pairs.append({"question": r.problems[rows[0]].question,
                              "chosen": _row(r, good[0]), "rejected": _row(r, bad[0])})
    stats = {"prompts": cfg.n_prompts, "pairs": len(pairs), "all_right": n_all_right,
             "all_wrong": n_all_wrong, "generated_tokens": generated}
    return pairs, stats


def _row(r, i):
    keep = r.attention_mask[i].bool()
    ids = r.input_ids[i][keep].cpu()
    n_completion = int(r.completion_mask[i].sum())
    return {"ids": ids, "n_completion": n_completion, "text": r.texts[i]}


def collate(rows, pad_id):
    """Left-pad token sequences into a batch, with a mask over each completion."""
    L = max(len(x["ids"]) for x in rows)
    ids = torch.full((len(rows), L), pad_id)
    attn = torch.zeros((len(rows), L), dtype=torch.long)
    comp = torch.zeros((len(rows), L), dtype=torch.long)
    for i, x in enumerate(rows):
        n = len(x["ids"])
        ids[i, L - n:] = x["ids"]
        attn[i, L - n:] = 1
        comp[i, L - x["n_completion"]:] = 1
    return ids.to(llm.DEVICE), attn.to(llm.DEVICE), comp.to(llm.DEVICE)


def sequence_logprob(model, ids, attn, comp):
    """log π(completion | prompt): the sum of the completion's token log-probs."""
    return (llm.token_logprobs(model, ids, attn) * comp).sum(1)


@torch.no_grad()
def reference_logprobs(ref_model, pairs, pad_id, micro_batch=8):
    """The reference model never changes, so compute its log-probs once, up front."""
    for i in range(0, len(pairs), micro_batch):
        chunk = pairs[i:i + micro_batch]
        rows = [p["chosen"] for p in chunk] + [p["rejected"] for p in chunk]
        lp = sequence_logprob(ref_model, *collate(rows, pad_id))
        for p, c, r in zip(chunk, lp[:len(chunk)], lp[len(chunk):]):
            p["ref_chosen"], p["ref_rejected"] = c.item(), r.item()


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    chosen_reward = beta * (pi_chosen - ref_chosen)  # the "implicit reward" of each reply
    rejected_reward = beta * (pi_rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin)
    return loss, chosen_reward, rejected_reward


def train(model, ref_model, tok, pairs, cfg, callback=None):
    torch.manual_seed(cfg.seed)
    reference_logprobs(ref_model, pairs, tok.pad_token_id, cfg.micro_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    history = []
    order = list(range(len(pairs)))
    rng = random.Random(cfg.seed)
    model.train()
    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), cfg.pairs_per_step):
            batch = [pairs[i] for i in order[start:start + cfg.pairs_per_step]]
            opt.zero_grad()
            totals = {"loss": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0, "pref_accuracy": 0.0}
            for mb in range(0, len(batch), cfg.micro_batch):
                chunk = batch[mb:mb + cfg.micro_batch]
                rows = [p["chosen"] for p in chunk] + [p["rejected"] for p in chunk]
                lp = sequence_logprob(model, *collate(rows, tok.pad_token_id))
                ref_c = torch.tensor([p["ref_chosen"] for p in chunk], device=lp.device)
                ref_r = torch.tensor([p["ref_rejected"] for p in chunk], device=lp.device)
                loss, cr, rr = dpo_loss(lp[:len(chunk)], lp[len(chunk):], ref_c, ref_r, cfg.beta)
                (loss.sum() / len(batch)).backward()
                totals["loss"] += loss.sum().item()
                totals["chosen_reward"] += cr.sum().item()
                totals["rejected_reward"] += rr.sum().item()
                totals["pref_accuracy"] += (cr > rr).float().sum().item()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            row = {"step": len(history), "epoch": epoch, "grad_norm": grad_norm.item()}
            row.update({k: v / len(batch) for k, v in totals.items()})
            row["margin"] = row["chosen_reward"] - row["rejected_reward"]
            history.append(row)
            if callback:
                callback(history)
    return history
