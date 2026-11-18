"""Юнит-тесты мониторинга: PSI/KS, доля дрейфа, триггер retrain, алерты."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lab4.monitor import (psi, ks, drift_report, drift_score,
                          check_retrain_needed, send_alert, ALERT_LOG,
                          RETRAIN_DRIFT, RETRAIN_AUC_DROP)


class TestPSI(unittest.TestCase):
    def test_same_distribution_low(self):
        a = np.random.default_rng(0).normal(size=5000)
        b = np.random.default_rng(1).normal(size=5000)
        self.assertLess(psi(a, b), 0.1)

    def test_shift_high(self):
        a = np.random.default_rng(0).normal(size=5000)
        b = np.random.default_rng(1).normal(loc=1.5, size=5000)
        self.assertGreater(psi(a, b), 0.25)

    def test_ks_detects_shift(self):
        a = np.random.default_rng(0).normal(size=1000)
        b = np.random.default_rng(1).normal(loc=1.0, size=1000)
        stat, pval = ks(a, b)
        self.assertGreater(stat, 0.2)
        self.assertLess(pval, 1e-10)


class TestDriftScore(unittest.TestCase):
    def test_fraction_of_drifted_columns(self):
        rng = np.random.default_rng(3)
        ref = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000),
                            "c": rng.normal(size=2000)})
        cur = ref.copy()
        cur["b"] = cur["b"] + 2.0  # одна колонка из трёх уплыла
        rep = drift_report(ref, cur, ["a", "b", "c"])
        self.assertEqual(int(rep["drifted"].sum()), 1)
        self.assertAlmostEqual(drift_score(rep), 1 / 3, places=6)

    def test_clean_window_below_alert(self):
        rng = np.random.default_rng(4)
        ref = pd.DataFrame({"a": rng.normal(size=3000)})
        cur = pd.DataFrame({"a": rng.normal(size=3000)})
        self.assertLess(drift_score(drift_report(ref, cur, ["a"])), 0.1)


class TestRetrain(unittest.TestCase):
    def test_drift_trigger(self):
        need, why = check_retrain_needed(RETRAIN_DRIFT + 0.05, 0.0, 0.0)
        self.assertTrue(need)
        self.assertIn("drift", why)

    def test_auc_drop_trigger(self):
        need, _ = check_retrain_needed(0.0, RETRAIN_AUC_DROP + 0.05, 0.0)
        self.assertTrue(need)

    def test_weekly_trigger(self):
        need, why = check_retrain_needed(0.0, 0.0, 7.0)
        self.assertTrue(need)
        self.assertIn("retrain", why)

    def test_quiet_case(self):
        need, _ = check_retrain_needed(0.05, 0.02, 2.0)
        self.assertFalse(need)


class TestAlerts(unittest.TestCase):
    def test_alert_written(self):
        send_alert("unit-test-маркер")
        text = ALERT_LOG.read_text(encoding="utf-8")
        self.assertIn("unit-test-маркер", text)


if __name__ == "__main__":
    unittest.main()
