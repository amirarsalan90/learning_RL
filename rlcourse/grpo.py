"""GRPO (Group Relative Policy Optimization) for an LLM.

GRPO = REINFORCE with two changes:
  * the baseline is the average reward of the *other samples for the same prompt*
    (a "group"), scaled by the group's standard deviation. No learned critic needed.
  * each batch of samples is reused for several gradient steps. After the first step
    the samples come from an older version of the model, so each token's loss is
    weighted by the probability ratio π_new / π_old, and that ratio is clipped to
    [1 - eps, 1 + eps] so one batch can't push the model too far (the PPO trick).
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
    updates_per_batch: int = 2
    clip_eps: float = 0.2
    micro_batch: int = 8
    max_grad_norm: float = 1.0
    difficulty: str = "medium"
    seed: int = 0


def advantages(rewards, group, state, cfg):
    """How much better each completion did than the other completions for the same prompt."""
    n_groups = int(group.max()) + 1
    g = rewards.view(n_groups, -1)  # rows of one group are adjacent
    adv = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)
    state["zero_variance_groups"] = (g.std(1) == 0).float().mean().item()
    return adv.view(-1)


def kl_penalty(logp, ref_logp):
    """Per-token estimate of KL(π || π_ref): always >= 0, and 0 where the two models agree."""
    d = ref_logp - logp
    return d.exp() - d - 1


def policy_loss(logp, old_logp, ref_logp, adv, mask, cfg, n_tokens):
    ratio = (logp - old_logp).exp()  # π_new / π_old for each token; exactly 1 on the first update
    unclipped = ratio * adv[:, None]
    clipped = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[:, None]
    pg = -torch.min(unclipped, clipped)  # pessimistic: take the smaller improvement
    kl = kl_penalty(logp, ref_logp)
    per_token = pg + cfg.kl_coef * kl
    loss = (per_token * mask).sum() / n_tokens
    stats = {"kl": (kl * mask).sum().item(),
             "clip_frac": ((unclipped != clipped).float() * mask).sum().item(),
             "ratio_dev": ((ratio - 1).abs() * mask).sum().item()}
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
        old_logp = llm.batched_logprobs(model, r, cfg.micro_batch)  # π_old: the model that sampled
        mask = r.completion_mask.float()
        n_tokens = mask.sum()

        # 3-5. several gradient steps on the same batch
        model.train()
        for update in range(cfg.updates_per_batch):
            opt.zero_grad()
            totals = {}
            for idx in llm.micro_batches(len(r), cfg.micro_batch):
                logp = llm.token_logprobs(model, r.input_ids[idx], r.attention_mask[idx])
                loss, stats = policy_loss(logp, old_logp[idx], ref_logp[idx], adv[idx], mask[idx], cfg, n_tokens)
                loss.backward()
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0.0) + v
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

        # log (stats are from the last update on this batch)
        row = {"step": step, **llm.rollout_stats(r), "grad_norm": grad_norm.item(),
               "zero_variance_groups": state["zero_variance_groups"], "adv_abs": adv.abs().mean().item()}
        row.update({k: v / n_tokens.item() for k, v in totals.items()})
        history.append(row)
        samples.append({"step": step, "question": r.problems[0].question,
                        "completions": [{"text": t, "reward": rw.item(), "advantage": a.item()}
                                        for t, rw, a in zip(r.texts[:cfg.samples_per_prompt], r.rewards, adv)]})
        if callback:
            callback(history)
    return history, samples
