"""PPO (Proximal Policy Optimization) for an LLM, in the classic RLHF form.

PPO = GRPO's clipped update, but the baseline comes from a learned **critic** (a value
model) instead of a group of samples:
  * the critic reads the prompt and the completion so far and predicts the final
    reward at every token, V(s_t) ("how likely is this to end up correct?")
  * per-token advantages come from how those predictions change as tokens are added
    (GAE, Generalized Advantage Estimation), so each token gets its own credit,
    instead of every token in a completion sharing one number
  * the KL penalty to the reference model is subtracted from the reward, token by token
  * one sample per prompt is enough: no groups needed
The price is a second model to train and hold in memory.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModel

from . import llm, task


@dataclass
class Config:
    steps: int = 80
    prompts_per_step: int = 64
    samples_per_prompt: int = 1
    max_new_tokens: int = 200
    temperature: float = 1.0
    lr: float = 2e-6
    critic_lr: float = 1e-5
    kl_coef: float = 0.05
    updates_per_batch: int = 2
    clip_eps: float = 0.2
    value_clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    micro_batch: int = 8
    max_grad_norm: float = 1.0
    difficulty: str = "medium"
    seed: int = 0


class Critic(torch.nn.Module):
    """The same transformer body as the policy, with a one-number head instead of the vocabulary head."""

    def __init__(self, name=llm.MODEL_NAME):
        super().__init__()
        self.body = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = torch.nn.Linear(self.body.config.hidden_size, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        """values[:, t] = V(state before token t): the predicted final reward, given tokens < t."""
        with llm.autocast():
            h = self.body(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=llm.position_ids(attention_mask)).last_hidden_state
        v = self.head(h.float()).squeeze(-1)
        return torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)


def token_rewards(rewards, logp, ref_logp, mask, cfg):
    """Reward at every token: -kl_coef * (log π - log π_ref), plus the real reward on the last token."""
    r = -cfg.kl_coef * (logp - ref_logp) * mask
    last = mask.cumsum(1).argmax(1)  # index of each row's last completion token
    r[torch.arange(len(r)), last] += rewards
    return r


def gae(token_r, values, mask, cfg):
    """Generalized Advantage Estimation, walking backwards through each completion.

    delta_t = r_t + gamma * V_{t+1} - V_t    (was token t better than the critic expected?)
    A_t     = delta_t + gamma * lam * A_{t+1}
    """
    adv = torch.zeros_like(values)
    running = torch.zeros_like(values[:, 0])
    L = values.shape[1]
    for t in reversed(range(L)):
        next_mask = mask[:, t + 1] if t + 1 < L else torch.zeros_like(mask[:, 0])
        next_v = values[:, t + 1] * next_mask if t + 1 < L else torch.zeros_like(running)
        delta = token_r[:, t] + cfg.gamma * next_v - values[:, t]
        running = (delta + cfg.gamma * cfg.lam * running * next_mask) * mask[:, t]
        adv[:, t] = running
    returns = adv + values  # what the critic should have predicted
    return adv, returns


def whiten(x, mask):
    mean = (x * mask).sum() / mask.sum()
    var = ((x - mean) ** 2 * mask).sum() / mask.sum()
    return (x - mean) / (var.sqrt() + 1e-8) * mask


def policy_loss(logp, old_logp, adv, mask, cfg, n_tokens):
    ratio = (logp - old_logp).exp()
    unclipped = ratio * adv  # adv is per token now: shape (batch, seq_len)
    clipped = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    pg = -torch.min(unclipped, clipped)
    loss = (pg * mask).sum() / n_tokens
    stats = {"clip_frac": ((unclipped != clipped).float() * mask).sum().item(),
             "ratio_dev": ((ratio - 1).abs() * mask).sum().item()}
    return loss, stats


def value_loss(values, old_values, returns, mask, cfg, n_tokens):
    clipped = old_values + (values - old_values).clamp(-cfg.value_clip, cfg.value_clip)
    loss = 0.5 * torch.max((values - returns) ** 2, (clipped - returns) ** 2)
    return (loss * mask).sum() / n_tokens


def explained_variance(values, returns, mask):
    v, r = values[mask.bool()], returns[mask.bool()]
    return (1 - (r - v).var() / (r.var() + 1e-8)).item()


def train(model, ref_model, critic, tok, cfg, callback=None):
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    stream = task.TrainStream(cfg.difficulty, cfg.seed)
    history, samples = [], []
    for step in range(cfg.steps):
        # 1-2. sample and score
        r = llm.generate(model, tok, stream.next(cfg.prompts_per_step), cfg.samples_per_prompt,
                         cfg.max_new_tokens, cfg.temperature)
        mask = r.completion_mask.float()
        n_tokens = mask.sum()
        ref_logp = llm.batched_logprobs(ref_model, r, cfg.micro_batch)
        old_logp = llm.batched_logprobs(model, r, cfg.micro_batch)
        with torch.no_grad():
            old_values = torch.cat([critic(r.input_ids[i], r.attention_mask[i])
                                    for i in llm.micro_batches(len(r), cfg.micro_batch)]) * mask

        # 3. per-token advantages from the critic
        token_r = token_rewards(r.rewards.to(llm.DEVICE), old_logp, ref_logp, mask, cfg)
        adv, returns = gae(token_r, old_values, mask, cfg)
        adv = whiten(adv, mask)

        # 4-5. several clipped updates of the policy and the critic on the same batch
        model.train()
        critic.train()
        for update in range(cfg.updates_per_batch):
            opt.zero_grad()
            critic_opt.zero_grad()
            totals = {"value_loss": 0.0}
            for idx in llm.micro_batches(len(r), cfg.micro_batch):
                logp = llm.token_logprobs(model, r.input_ids[idx], r.attention_mask[idx])
                loss, stats = policy_loss(logp, old_logp[idx], adv[idx], mask[idx], cfg, n_tokens)
                loss.backward()
                values = critic(r.input_ids[idx], r.attention_mask[idx])
                v_loss = value_loss(values, old_values[idx], returns[idx], mask[idx], cfg, n_tokens)
                v_loss.backward()
                totals["value_loss"] += v_loss.item() * n_tokens.item()
                for k, v in stats.items():
                    totals[k] = totals.get(k, 0.0) + v
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            opt.step()
            critic_opt.step()

        # log
        kl = ((old_logp - ref_logp) * mask).sum() / n_tokens
        row = {"step": step, **llm.rollout_stats(r), "grad_norm": grad_norm.item(), "kl": kl.item(),
               "explained_variance": explained_variance(old_values, returns, mask),
               "value_at_start": (old_values * (mask.cumsum(1) == 1)).sum().item() / len(r)}
        row.update({k: v / n_tokens.item() for k, v in totals.items()})
        history.append(row)
        samples.append({"step": step, "question": r.problems[0].question,
                        "completions": [{"text": r.texts[0], "reward": r.rewards[0].item()}]})
        if callback:
            callback(history)
    return history, samples
