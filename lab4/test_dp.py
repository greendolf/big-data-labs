"""Юнит-тесты DP-механизма (без тяжёлых данных)."""
import unittest

import numpy as np

from lab4.dp import rdp_epsilon, sigma_for_epsilon, train_logreg_dp, predict_proba


class TestRDP(unittest.TestCase):
    def test_monotone_decreasing_in_sigma(self):
        sigmas = [0.3, 0.5, 1.0, 2.0]
        eps = [rdp_epsilon(s, steps=1000, delta=1e-6) for s in sigmas]
        self.assertTrue(all(a > b for a, b in zip(eps, eps[1:])), eps)

    def test_linear_in_steps(self):
        e1 = rdp_epsilon(1.0, steps=100, delta=1e-6)
        e10 = rdp_epsilon(1.0, steps=1000, delta=1e-6)
        self.assertGreater(e10, 5 * e1)  # композиция растёт почти линейно

    def test_sigma_inverse_matches_budget(self):
        steps, delta, target = 500, 1e-6, 2.0
        s = sigma_for_epsilon(target, steps, delta)
        self.assertLessEqual(rdp_epsilon(s, steps, delta), target + 1e-6)
        self.assertGreater(rdp_epsilon(s * 0.99, steps, delta), target)

    def test_infinite_sigma_zero_noise(self):
        self.assertEqual(rdp_epsilon(0.0, 100, 1e-6), float("inf"))


class TestTrain(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 4000
        self.X = rng.normal(size=(n, 4))
        self.y = (self.X[:, 0] + 0.5 * self.X[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)

    def test_baseline_learns(self):
        r = train_logreg_dp(self.X, self.y, steps=600, sigma=0.0, epsilon=None)
        p = predict_proba(r.weights, r.bias, self.X)
        auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(self.y, p)
        self.assertGreater(auc, 0.85)

    def test_dp_noise_deterministic_with_seed(self):
        kw = dict(steps=200, epsilon=2.0, seed=7)
        a = train_logreg_dp(self.X, self.y, **kw)
        b = train_logreg_dp(self.X, self.y, **kw)
        np.testing.assert_allclose(a.weights, b.weights)
        self.assertLessEqual(a.epsilon, 2.0 + 1e-6)

    def test_privacy_cost_monotone(self):
        from sklearn.metrics import roc_auc_score

        aucs = []
        for eps in (8.0, 2.0):     # оба достижимы при 600 шагов (ε_min≈1.3)
            r = train_logreg_dp(self.X, self.y, steps=600, epsilon=eps, seed=1)
            aucs.append(roc_auc_score(self.y, predict_proba(r.weights, r.bias, self.X)))
        # меньший бюджет не должен быть КАЧЕСТВЕННЕЕ (с запасом на шум сидов)
        self.assertLess(aucs[1], aucs[0] + 0.02)

    def test_infeasible_epsilon_raises(self):
        from lab4.dp import epsilon_min
        with self.assertRaises(ValueError):
            train_logreg_dp(self.X, self.y, steps=3000, epsilon=0.5, seed=1)
        self.assertGreater(epsilon_min(3000, 1e-6), 2.0)  # ε<2 при 3000 шагов недостижим


if __name__ == "__main__":
    unittest.main()
