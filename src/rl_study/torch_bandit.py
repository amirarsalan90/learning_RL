"""Autograd counterpart to the explicit tabular REINFORCE update."""

import torch


class TorchAgent:
    def __init__(self):
        # Float64 makes numerical comparisons to the explicit backend meaningful.
        self.logits = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
        self.values = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def probabilities(self):
        with torch.no_grad():
            return self.logits.softmax(dim=-1).tolist()

    def baseline_values(self):
        return self.values.detach().tolist()

    def update(self, batch, baseline, policy_lr, value_lr, learn_value):
        contexts = torch.tensor([c for c, _, _ in batch], dtype=torch.long)
        actions = torch.tensor([a for _, a, _ in batch], dtype=torch.long)
        rewards = torch.tensor([r for _, _, r in batch], dtype=torch.float64)
        baseline_tensor = torch.tensor(baseline, dtype=torch.float64)[contexts]
        advantages = (rewards - baseline_tensor).detach()
        log_probs = self.logits[contexts].log_softmax(dim=-1)
        selected_log_probs = log_probs.gather(1, actions[:, None]).squeeze(1)
        policy_loss = -(advantages * selected_log_probs).mean()
        policy_loss.backward()
        if learn_value:
            value_loss = 0.5 * (self.values[contexts] - rewards.detach()).square().mean()
            value_loss.backward()
        with torch.no_grad():
            self.logits -= policy_lr * self.logits.grad
            if learn_value:
                self.values -= value_lr * self.values.grad
        self.logits.grad = None
        self.values.grad = None
