import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rl_study.bandit import (BASELINES, Config, PythonAgent, batch_gradient, evaluate,
                             exact_estimator_stats, exact_gradient, frozen_variance_experiment,
                             sample_batch, softmax, train)


class BanditTests(unittest.TestCase):
    def test_softmax_is_stable(self):
        probabilities = softmax([10000, 10001, 9999])
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertTrue(all(math.isfinite(p) for p in probabilities))

    def test_advantage_sign_changes_action_probability(self):
        for reward, baseline, direction in ((1.0, 0.0, 1), (0.0, 1.0, -1)):
            agent = PythonAgent()
            before = agent.probabilities()[0][1]
            agent.update([(0, 1, reward)], [baseline] * 3, 0.1, 0.1, False)
            self.assertGreater(direction * (agent.probabilities()[0][1] - before), 0)
            self.assertEqual(agent.values, [0.0] * 3)

    def test_context_baselines_leave_exact_expected_gradient_unchanged(self):
        probabilities = [softmax(row) for row in ((0.4, -0.2, 0.7), (-1, 2, 0), (1, 0.4, 0.3))]
        target = exact_gradient(probabilities)
        for baseline in ([0, 0, 0], [0.4, 0.4, 0.4], [0.2, 0.8, -0.1]):
            mean, _ = exact_estimator_stats(probabilities, baseline)
            for c in range(3):
                for a in range(3):
                    self.assertAlmostEqual(mean[c][a], target[c][a], places=12)

    def test_exact_gradient_matches_finite_difference(self):
        logits = [[0.4, -0.2, 0.7], [-1, 2, 0], [1, 0.4, 0.3]]
        target = exact_gradient([softmax(row) for row in logits])
        epsilon = 1e-5
        for c in range(3):
            for a in range(3):
                high, low = [row[:] for row in logits], [row[:] for row in logits]
                high[c][a] += epsilon
                low[c][a] -= epsilon
                difference = (evaluate([softmax(r) for r in high])["expected_reward"] -
                              evaluate([softmax(r) for r in low])["expected_reward"]) / (2 * epsilon)
                self.assertAlmostEqual(difference, target[c][a], places=8)

    def test_monte_carlo_and_exact_variance_agree(self):
        results = frozen_variance_experiment(batch_size=32, repetitions=1500)
        for result in results:
            self.assertLess(result["max_gradient_error"], 0.003)
            ratio = result["empirical_variance"] / result["exact_variance"]
            self.assertLess(abs(ratio - 1), 0.1)
        self.assertGreater(results[0]["exact_variance"], results[1]["exact_variance"])
        self.assertGreater(results[1]["exact_variance"], results[2]["exact_variance"])

    def test_value_update_is_separate_from_policy_update(self):
        batch = [(0, 0, 1), (0, 1, 0), (1, 2, 1)]
        agents = [PythonAgent(), PythonAgent()]
        for agent, learn in zip(agents, (False, True)):
            agent.update(batch, [0.2, 0.3, 0.4], 0.1, 0.3, learn)
        self.assertEqual(agents[0].logits, agents[1].logits)
        self.assertEqual(agents[0].values, [0.0] * 3)
        self.assertAlmostEqual(agents[1].values[0], 0.1)
        self.assertAlmostEqual(agents[1].values[1], 0.1)

    def test_all_methods_learn_and_are_reproducible(self):
        config = Config(steps=250, batch_size=32, seeds=(42,))
        for method in BASELINES:
            rows = train(config, method, 42)
            self.assertEqual(rows, train(config, method, 42))
            self.assertGreater(rows[-1]["expected_reward"], rows[0]["expected_reward"] + 0.2)
            self.assertGreater(rows[-1]["optimal_action_probability"], 0.85)
            self.assertEqual(rows[-1]["samples"], 250 * 32)

    def test_invalid_configuration(self):
        for config in (Config(steps=0), Config(batch_size=-1), Config(seeds=()),
                       Config(seeds=(0, 0)), Config(policy_lr=float("nan")), Config(running_decay=1)):
            with self.assertRaises(ValueError):
                config.validate()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency is not installed")
    def test_autograd_backend_matches_explicit_update(self):
        from rl_study.torch_bandit import TorchAgent
        python, torch_agent = PythonAgent(), TorchAgent()
        rng = random.Random(7)
        for _ in range(20):
            batch = sample_batch(python.probabilities(), rng, 32)
            baseline = python.baseline_values()
            for agent in (python, torch_agent):
                agent.update(batch, baseline, 0.5, 0.5, True)
            for row1, row2 in zip(python.probabilities(), torch_agent.probabilities()):
                for x, y in zip(row1, row2):
                    self.assertAlmostEqual(x, y, places=12)
            for x, y in zip(python.values, torch_agent.baseline_values()):
                self.assertAlmostEqual(x, y, places=12)


if __name__ == "__main__":
    unittest.main()
