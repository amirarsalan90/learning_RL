"""Contextual REINFORCE, with explicit softmax gradients and no dependencies.

Each context has its own three logits and (for the learned baseline) one value.
This is a tabular policy/critic, deliberately simpler than a neural network.
All gradient functions return gradients of expected reward (gradient ASCENT).
"""

from dataclasses import asdict, dataclass
import math
import random


REWARD_PROBS = ((0.85, 0.50, 0.30), (0.05, 0.75, 0.10), (0.25, 0.10, 0.90))
BASELINES = ("none", "running", "learned")


def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    total = sum(weights)
    return [w / total for w in weights]


def zero_matrix():
    return [[0.0] * 3 for _ in range(3)]


def score_gradient(probabilities, context, action, advantage):
    """A * grad(log pi(a|c)); d log softmax(a) / dz_j = 1[a=j]-pi(j)."""
    gradient = zero_matrix()
    gradient[context] = [
        advantage * ((1.0 if j == action else 0.0) - p)
        for j, p in enumerate(probabilities[context])
    ]
    return gradient


def exact_gradient(probabilities):
    """Analytic gradient of J = mean_c sum_a pi(a|c) * E[r|c,a]."""
    gradient = zero_matrix()
    for context in range(3):
        expected = sum(p * r for p, r in zip(probabilities[context], REWARD_PROBS[context]))
        gradient[context] = [
            p * (r - expected) / 3.0
            for p, r in zip(probabilities[context], REWARD_PROBS[context])
        ]
    return gradient


def exact_estimator_stats(probabilities, baseline, batch_size=1):
    """Enumerate contexts, actions, and binary rewards at a FROZEN policy.

    Returns E[g] and trace(Cov(g_batch)); an IID batch mean divides variance by B.
    Baselines must depend only on context, never the current sampled action/reward.
    This measures gradient noise, not variability of reward across training time.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mean = zero_matrix()
    squared_norm = 0.0
    for context in range(3):
        for action in range(3):
            success = REWARD_PROBS[context][action]
            for reward, reward_probability in ((0.0, 1.0 - success), (1.0, success)):
                probability = probabilities[context][action] * reward_probability / 3.0
                gradient = score_gradient(probabilities, context, action, reward - baseline[context])
                for c in range(3):
                    for a in range(3):
                        mean[c][a] += probability * gradient[c][a]
                        squared_norm += probability * gradient[c][a] ** 2
    mean_squared_norm = sum(v * v for row in mean for v in row)
    return mean, max(0.0, squared_norm - mean_squared_norm) / batch_size


def sample_batch(probabilities, rng, batch_size):
    batch = []
    for _ in range(batch_size):
        context = rng.randrange(3)
        action = rng.choices(range(3), weights=probabilities[context], k=1)[0]
        reward = float(rng.random() < REWARD_PROBS[context][action])
        batch.append((context, action, reward))
    return batch


def batch_gradient(probabilities, batch, baseline):
    if not batch:
        raise ValueError("batch must be nonempty")
    gradient = zero_matrix()
    for context, action, reward in batch:
        sample = score_gradient(probabilities, context, action, reward - baseline[context])
        for a in range(3):
            gradient[context][a] += sample[context][a] / len(batch)
    return gradient


class PythonAgent:
    def __init__(self):
        self.logits = zero_matrix()
        self.values = [0.0] * 3

    def probabilities(self):
        return [softmax(row) for row in self.logits]

    def baseline_values(self):
        return list(self.values)

    def update(self, batch, baseline, policy_lr, value_lr, learn_value):
        gradient = batch_gradient(self.probabilities(), batch, baseline)
        for context in range(3):
            for action in range(3):
                self.logits[context][action] += policy_lr * gradient[context][action]
        if learn_value:
            # Gradient descent on 0.5 * mean((V(context) - reward)^2).
            old_values = list(self.values)
            for context, _, reward in batch:
                self.values[context] += value_lr * (reward - old_values[context]) / len(batch)


@dataclass(frozen=True)
class Config:
    steps: int = 1000
    batch_size: int = 64
    policy_lr: float = 0.5
    value_lr: float = 0.5
    running_decay: float = 0.95
    log_every: int = 20
    seeds: tuple = (0, 1, 2, 3, 4)
    backend: str = "python"

    def validate(self):
        if self.steps <= 0 or self.batch_size <= 0 or self.log_every <= 0:
            raise ValueError("steps, batch_size, and log_every must be positive")
        if not all(math.isfinite(x) and x > 0 for x in (self.policy_lr, self.value_lr)):
            raise ValueError("learning rates must be finite and positive")
        if not 0 <= self.running_decay < 1:
            raise ValueError("running_decay must be in [0, 1)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("provide at least one seed, with no duplicates")
        if self.backend not in ("python", "torch"):
            raise ValueError("backend must be python or torch")


def evaluate(probabilities):
    expected_reward = sum(
        p * r for probs, rewards in zip(probabilities, REWARD_PROBS)
        for p, r in zip(probs, rewards)
    ) / 3.0
    optimal_action_probability = sum(
        probs[max(range(3), key=lambda a: REWARD_PROBS[c][a])]
        for c, probs in enumerate(probabilities)
    ) / 3.0
    entropy = -sum(p * math.log(max(p, 1e-300)) for row in probabilities for p in row) / 3.0
    return dict(expected_reward=expected_reward,
                optimal_action_probability=optimal_action_probability, entropy=entropy)


def train(config, baseline_kind, seed):
    config.validate()
    if baseline_kind not in BASELINES:
        raise ValueError(f"unknown baseline: {baseline_kind}")
    if config.backend == "torch":
        from .torch_bandit import TorchAgent
        agent = TorchAgent()
    else:
        agent = PythonAgent()
    rng = random.Random(seed)
    running = 0.0
    rows = []
    for step in range(config.steps + 1):
        probabilities = agent.probabilities()
        baseline = ([0.0] * 3 if baseline_kind == "none" else
                    [running] * 3 if baseline_kind == "running" else agent.baseline_values())
        if step % config.log_every == 0 or step == config.steps:
            _, variance = exact_estimator_stats(probabilities, baseline, config.batch_size)
            rows.append(dict(method=baseline_kind, seed=seed, step=step,
                             samples=step * config.batch_size,
                             gradient_variance=variance,
                             baseline_mean=sum(baseline) / 3,
                             **evaluate(probabilities)))
        if step == config.steps:
            break
        batch = sample_batch(probabilities, rng, config.batch_size)
        agent.update(batch, baseline, config.policy_lr, config.value_lr, baseline_kind == "learned")
        # Update AFTER the policy uses the previous baseline. Including a sample's
        # own reward in its baseline would bias the plain REINFORCE estimator.
        if baseline_kind == "running":
            reward_mean = sum(r for _, _, r in batch) / len(batch)
            running = config.running_decay * running + (1 - config.running_decay) * reward_mean
    return rows


def frozen_variance_experiment(batch_size=64, repetitions=2000, seed=123):
    """Compare baseline estimators using identical batches at a uniform policy."""
    if batch_size <= 0 or repetitions < 2:
        raise ValueError("batch_size must be positive and repetitions at least two")
    probabilities = [softmax(row) for row in zero_matrix()]
    context_values = [sum(row) / 3.0 for row in REWARD_PROBS]
    baselines = {"none": [0.0] * 3,
                 "running": [sum(context_values) / 3.0] * 3,
                 "learned": context_values}
    rng = random.Random(seed)
    samples = {name: [] for name in BASELINES}
    for _ in range(repetitions):
        batch = sample_batch(probabilities, rng, batch_size)
        for name, baseline in baselines.items():
            samples[name].append([x for row in batch_gradient(probabilities, batch, baseline) for x in row])
    target = [x for row in exact_gradient(probabilities) for x in row]
    result = []
    for name, baseline in baselines.items():
        means = [sum(g[j] for g in samples[name]) / repetitions for j in range(9)]
        empirical_variance = sum(
            sum((g[j] - means[j]) ** 2 for g in samples[name]) / (repetitions - 1)
            for j in range(9)
        )
        _, exact_variance = exact_estimator_stats(probabilities, baseline, batch_size)
        result.append(dict(method=name, exact_variance=exact_variance,
                           empirical_variance=empirical_variance,
                           max_gradient_error=max(abs(a - b) for a, b in zip(means, target))))
    return result


def config_dict(config):
    return {**asdict(config), "reward_probabilities": REWARD_PROBS,
            "context_distribution": "uniform", "policy": "tabular softmax",
            "critic_loss": "0.5 * mean squared error", "optimizer": "SGD"}
