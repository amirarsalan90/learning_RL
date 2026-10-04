#!/usr/bin/env python3
"""Run all baseline variants and write reproducible local experiment artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_study.bandit import BASELINES, Config, REWARD_PROBS, config_dict, frozen_variance_experiment, train
from rl_study.report import learning_curves_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy-lr", type=float, default=0.5)
    parser.add_argument("--value-lr", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--backend", choices=["python", "torch"], default="python")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = Config(steps=args.steps, batch_size=args.batch_size,
                    policy_lr=args.policy_lr, value_lr=args.value_lr,
                    seeds=tuple(args.seeds), backend=args.backend)
    try:
        config.validate()
        if args.backend == "torch":
            import torch  # Check before creating a run directory.
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    # Never overwrite an experiment. A supplied output directory must be new.
    output = args.output or Path(__file__).resolve().parents[1] / "runs" / datetime.now(timezone.utc).strftime("bandit-%Y%m%dT%H%M%S-%fZ")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists: {output}; choose a new directory")
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    metadata = {**config_dict(config), "python": platform.python_version(),
                "packages": versions, "created_at": datetime.now(timezone.utc).isoformat()}
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for method in BASELINES:
        for seed in config.seeds:
            result = train(config, method, seed)
            rows.extend(result)
            print(f"{method:7s} seed={seed}: reward={result[-1]['expected_reward']:.4f}, "
                  f"optimal action={result[-1]['optimal_action_probability']:.4f}", flush=True)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    variance = frozen_variance_experiment(config.batch_size)
    (output / "frozen_variance.json").write_text(json.dumps(variance, indent=2) + "\n")
    (output / "learning_curves.svg").write_text(learning_curves_svg(rows))
    optimal_reward = sum(max(row) for row in REWARD_PROBS) / 3
    summary = ["# Bandit results", "", f"Analytic optimal expected reward: {optimal_reward:.4f}.", "",
               "| Baseline | Final expected reward (mean ± SD) | Optimal-action probability |", "|---|---:|---:|"]
    for method in BASELINES:
        final = [r for r in rows if r["method"] == method and r["step"] == config.steps]
        rewards = [r["expected_reward"] for r in final]
        sd = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        summary.append(f"| {method} | {statistics.mean(rewards):.4f} ± {sd:.4f} | "
                       f"{statistics.mean(r['optimal_action_probability'] for r in final):.4f} |")
    summary += ["", "## Frozen-policy variance", "",
                "At a uniform policy, the running/learned baselines below are their converged values,",
                "not their initial zero values. Identical sampled batches are used for every estimator.", "",
                "| Baseline | Exact variance | Empirical variance | Max gradient error |", "|---|---:|---:|---:|"]
    for item in variance:
        summary.append(f"| {item['method']} | {item['exact_variance']:.6f} | "
                       f"{item['empirical_variance']:.6f} | {item['max_gradient_error']:.6f} |")
    summary += ["", "Variance is the trace of the covariance of a batch-mean gradient.",
                "Learning-curve bands show one standard deviation across seeds, not confidence intervals.",
                "A learned value baseline is not guaranteed to minimize gradient variance at every policy.",
                "", "![Learning curves](learning_curves.svg)", ""]
    (output / "summary.md").write_text("\n".join(summary))
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
