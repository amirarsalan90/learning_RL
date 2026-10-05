"""The task every LLM notebook trains on: small arithmetic problems with a checkable answer.

The model must end its reply with <answer>NUMBER</answer>. The reward is 1 if that
number is right and 0 otherwise (including when the tag is missing). That's a
"verifiable reward": no learned reward model, no human labels, just a Python check,
the same kind of reward used to train reasoning models on math and code.
"""

import random
import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a careful math assistant. Think step by step, briefly. "
    "End your reply with the final answer in the form <answer>NUMBER</answer>."
)

ANSWER_RE = re.compile(r"<answer>\s*(-?[\d,]+)\s*</answer>")


@dataclass(frozen=True)
class Problem:
    question: str
    answer: int


def make_problem(rng, difficulty="medium"):
    if difficulty == "easy":  # a + b
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return Problem(f"What is {a} + {b}?", a + b)
    if difficulty == "medium":  # a * b + c
        a, b, c = rng.randint(11, 99), rng.randint(3, 9), rng.randint(10, 99)
        return Problem(f"What is {a} * {b} + {c}?", a * b + c)
    if difficulty == "hard":  # a * b - c * d
        a, b, c, d = (rng.randint(11, 99) for _ in range(4))
        return Problem(f"What is {a} * {b} - {c} * {d}?", a * b - c * d)
    raise ValueError(f"unknown difficulty {difficulty!r}")


def eval_problems(n=200, difficulty="medium"):
    """A fixed held-out set. Training problems never include these questions."""
    rng = random.Random(12345)
    return [make_problem(rng, difficulty) for _ in range(n)]


class TrainStream:
    """An endless stream of fresh training problems, disjoint from the eval set."""

    def __init__(self, difficulty="medium", seed=0):
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.held_out = {p.question for p in eval_problems(difficulty=difficulty)}

    def next(self, n):
        out = []
        while len(out) < n:
            p = make_problem(self.rng, self.difficulty)
            if p.question not in self.held_out:
                out.append(p)
        return out


def parse_answer(text):
    """The integer inside the last <answer>…</answer> tag, or None if there isn't one."""
    found = ANSWER_RE.findall(text)
    if not found:
        return None
    try:
        return int(found[-1].replace(",", ""))
    except ValueError:
        return None


def reward(text, answer):
    """1.0 for a well-formed, correct final answer, else 0.0."""
    return 1.0 if parse_answer(text) == answer else 0.0
