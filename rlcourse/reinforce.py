"""REINFORCE for an LLM, with a running-average baseline and a KL penalty.

One training step:
  1. sample completions for a batch of prompts from the current model
  2. score them: reward 1 if the answer is right, else 0
  3. advantage = reward - baseline, where the baseline is a running average of past rewards
  4. loss = -advantage * log π(token), for every token of every completion,
     plus a penalty for drifting away from the frozen starting model (the "reference")
  5. one gradient step, then throw the samples away (they're stale now)
"""

from dataclasses import dataclass

import torch

from . import llm, task


@dataclass
class Config:
    steps: int = 80
    prompts_per_step: int = 16
    samples_per_prompt: int = 4
    max_new_tokens: int = 200
    temperature: float = 1.0
    lr: float = 2e-6
    kl_coef: float = 0.02
    baseline_decay: float = 0.9
    micro_batch: int = 8
    max_grad_norm: float = 1.0
    difficulty: str = "medium"
    seed: int = 0


def advantages(rewards, group, state, cfg):
    """How much better each completion did than the running average of past rewards."""
    baseline = state.get("baseline", 0.0)
    state["baseline"] = cfg.baseline_decay * baseline + (1 - cfg.baseline_decay) * rewards.mean().item()
    return rewards - baseline


def kl_penalty(logp, ref_logp):
    """Per-token estimate of KL(π || π_ref): always >= 0, and 0 where the two models agree."""
    d = ref_logp - logp
    return d.exp() - d - 1


def policy_loss(logp, ref_logp, adv, mask, cfg, n_tokens):
    pg = -adv[:, None] * logp  # the stage-1 REINFORCE loss, applied to every token
    kl = kl_penalty(logp, ref_logp)
    per_token = pg + cfg.kl_coef * kl
    loss = (per_token * mask).sum() / n_tokens
    stats = {"kl": (kl * mask).sum().item()}
    return loss, stats


def train(model, ref_model, tok, cfg, callback=None):
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    stream = task.TrainStream(cfg.difficulty, cfg.seed)
    history, samples, state = [], [], {}
    for step in range(cfg.steps):
        # 1-2. sample and score
        r = llm.generate(model, tok, stream.next(cfg.prompts_per_step), cfg.samples_per_prompt,
                         cfg.max_new_tokens, cfg.temperature)
        adv = advantages(r.rewards, r.group, state, cfg).to(llm.DEVICE)
        ref_logp = llm.batched_logprobs(ref_model, r, cfg.micro_batch)
        mask = r.completion_mask.float()
        n_tokens = mask.sum()

        # 3-5. one gradient step on this batch
        model.train()
        opt.zero_grad()
        totals = {}
        for idx in llm.micro_batches(len(r), cfg.micro_batch):
            logp = llm.token_logprobs(model, r.input_ids[idx], r.attention_mask[idx])
            loss, stats = policy_loss(logp, ref_logp[idx], adv[idx], mask[idx], cfg, n_tokens)
            loss.backward()
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        # log
        row = {"step": step, **llm.rollout_stats(r), "grad_norm": grad_norm.item(),
               "baseline": state["baseline"], "adv_abs": adv.abs().mean().item()}
        row.update({k: v / n_tokens.item() for k, v in totals.items()})
        history.append(row)
        samples.append({"step": step, "question": r.problems[0].question,
                        "completions": [{"text": t, "reward": rw.item(), "advantage": a.item()}
                                        for t, rw, a in zip(r.texts[:cfg.samples_per_prompt], r.rewards, adv)]})
        if callback:
            callback(history)
    return history, samples
