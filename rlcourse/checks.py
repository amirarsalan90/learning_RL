"""Checks for the exercise cells.

Each check calls your function on a few inputs and compares it with what it should
produce. On success it prints a short confirmation; on failure it raises an
AssertionError whose message is a hint, not the answer.
"""

import torch


def _ok(name):
    print(f"✅ {name} looks right.")


def _fail(msg):
    raise AssertionError(msg)


# --------------------------------------------------------------------------- stage 1

_TEST_LOGITS = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.5])


def check_sample(sample):
    torch.manual_seed(123)
    logits = _TEST_LOGITS.clone().requires_grad_(True)
    out = sample(logits, 20_000)
    if not (isinstance(out, tuple) and len(out) == 2):
        _fail("sample should return a tuple (actions, log_probs).")
    actions, log_probs = out
    if actions.shape != (20_000,) or log_probs.shape != (20_000,):
        _fail(f"Expected both outputs to have shape (n,) = (20000,), got {tuple(actions.shape)} and {tuple(log_probs.shape)}.")
    if actions.dtype not in (torch.int64, torch.int32):
        _fail(f"actions should be integer arm indices, got dtype {actions.dtype}.")
    if actions.min() < 0 or actions.max() >= len(logits):
        _fail("actions should be arm indices in [0, n_arms).")
    if not log_probs.requires_grad:
        _fail(
            "log_probs has no gradient attached. The whole point is to backprop through "
            "log π(a), so compute it from `logits` with differentiable ops (log_softmax + indexing). "
            "Only the *sampling* step should be detached."
        )
    expected_lp = torch.log_softmax(logits.detach(), -1)[actions]
    if not torch.allclose(log_probs.detach(), expected_lp, atol=1e-5):
        _fail("log_probs[i] should equal log π(actions[i]) = log_softmax(logits)[actions[i]].")
    freq = torch.bincount(actions, minlength=len(logits)).float() / len(actions)
    probs = torch.softmax(logits.detach(), -1)
    if (freq - probs).abs().max() > 0.02:
        _fail(
            f"Sampled frequencies {freq.numpy().round(3)} don't match softmax(logits) "
            f"{probs.numpy().round(3)}. Are you sampling from the softmax probabilities, with replacement?"
        )
    _ok("sample")


def check_expected_reward(expected_reward, true_p):
    logits = _TEST_LOGITS.clone().requires_grad_(True)
    J = expected_reward(logits)
    if not torch.is_tensor(J) or J.numel() != 1:
        _fail("expected_reward should return a scalar tensor.")
    want = (torch.softmax(_TEST_LOGITS, -1) * true_p).sum()
    if not torch.allclose(J.detach().reshape(()), want, atol=1e-5):
        _fail(f"For logits {_TEST_LOGITS.tolist()} expected J = {want:.4f}, got {J.item():.4f}. J = Σ_a π(a) · p_a.")
    if not J.requires_grad:
        _fail("J must be differentiable with respect to logits, so autograd can give us the exact gradient.")
    _ok("expected_reward")


def check_reinforce_loss(reinforce_loss):
    torch.manual_seed(0)
    logits = torch.randn(5, requires_grad=True)
    actions = torch.tensor([0, 2, 2, 4, 1, 3])
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], requires_grad=True)
    log_probs = torch.log_softmax(logits, -1)[actions]
    loss = reinforce_loss(log_probs, rewards)
    if not torch.is_tensor(loss) or loss.numel() != 1:
        _fail("reinforce_loss should return a scalar tensor (we call .backward() on it).")
    want = -(rewards.detach() * log_probs.detach()).mean()
    if torch.allclose(loss.detach(), -want, atol=1e-5):
        _fail("Sign is flipped. Optimizers *minimize*, and we want to *maximize* reward-weighted log-probs.")
    if torch.allclose(loss.detach(), want * len(actions), atol=1e-5):
        _fail("Use the mean over the batch, not the sum, so the step size doesn't depend on batch size.")
    if not torch.allclose(loss.detach(), want, atol=1e-5):
        _fail(f"Expected {want.item():.4f}, got {loss.item():.4f}. The loss is -mean(reward_i · log π(a_i)).")
    loss.backward()
    if rewards.grad is not None and rewards.grad.abs().sum() > 0:
        _fail(
            "Gradient flowed into `rewards`. Rewards (and later, advantages) are constants from the "
            "environment: wrap them in .detach() so only log π gets differentiated."
        )
    _ok("reinforce_loss")


def check_train(train):
    finals = []
    for seed in range(3):
        _, history = train(steps=300, batch_size=16, lr=0.5, seed=seed)
        if len(history) != 300:
            _fail(f"history should have one entry per step (300), got {len(history)}.")
        finals.append(history[-1])
    if min(finals) < 0.7:
        _fail(
            f"Final expected rewards {[round(f, 3) for f in finals]}: should reach > 0.7 (optimum is 0.8). "
            "Check the order: sample → pull → loss → zero_grad → backward → step."
        )
    if history[0] > 0.6:
        _fail("history[0] is already high. Are you recording expected_reward(logits) *during* training?")
    _ok("train")


_REWARD_BATCHES = [
    torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
    torch.tensor([5.0, 6.0, 6.0, 5.0]),
    torch.tensor([0.3, -1.2, 2.5, 0.0, 0.7]),
]


def check_mean_baseline(mean_baseline):
    for r in _REWARD_BATCHES:
        got = mean_baseline(r.clone())
        want = r - r.mean()
        if got.shape != r.shape:
            _fail("Return one advantage per sample (same shape as rewards).")
        if not torch.allclose(got, want, atol=1e-5):
            _fail(f"For rewards {r.tolist()} expected {want.numpy().round(3).tolist()}, got {got.numpy().round(3).tolist()}.")
    _ok("mean_baseline")


def check_normalized(normalized):
    for r in _REWARD_BATCHES:
        got = normalized(r.clone())
        want = (r - r.mean()) / (r.std() + 1e-6)
        if not torch.allclose(got, want, atol=1e-4):
            _fail(f"For rewards {r.tolist()} expected {want.numpy().round(3).tolist()}, got {got.numpy().round(3).tolist()}.")
    same = normalized(torch.ones(4))
    if not torch.isfinite(same).all() or same.abs().max() > 1e-3:
        _fail("When all rewards are equal, std is 0. Add a small epsilon so the advantages are 0, not NaN.")
    _ok("normalized")


def check_loo_baseline(loo_baseline):
    for r in _REWARD_BATCHES:
        n = len(r)
        want = r - (r.sum() - r) / (n - 1)
        got = loo_baseline(r.clone())
        if not torch.allclose(got, want, atol=1e-5):
            _fail(
                f"For rewards {r.tolist()} expected {want.numpy().round(3).tolist()}, got {got.numpy().round(3).tolist()}. "
                "Sample i's baseline is the mean of the *other* n-1 rewards."
            )
    _ok("loo_baseline")
