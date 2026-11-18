"""Юнит-тесты части 8: чистые функции (без скачивания NAB-ряда)."""
import unittest

import numpy as np
import pandas as pd

from lab4.part8 import (evaluate_alarms, pinball_loss, mase, alarms_from_signals,
                        MonitorParams, INCIDENTS)


class TestMetrics(unittest.TestCase):
    def test_pinball_penalizes_underprediction(self):
        y = np.array([10.0, 10.0])
        low = np.array([8.0, 8.0])
        high = np.array([10.5, 10.5])
        self.assertGreater(pinball_loss(y, low, 0.9), pinball_loss(y, high, 0.9))

    def test_mase_naive_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        # lag-1 наивный: MAE = средний |diff| ⇒ MASE = 1
        self.assertAlmostEqual(mase(y[1:], y[:-1], y), 1.0, places=6)
        # константный прогноз =1: MAE=(1+2+3)/3=2, знаменатель 1 ⇒ MASE=2
        self.assertAlmostEqual(mase(y[1:], np.repeat(y[0], 3), y), 2.0, places=6)


class TestAlarmEval(unittest.TestCase):
    def test_lead_time_and_window(self):
        a0, a1 = INCIDENTS["accident_1"]
        alarms = pd.DatetimeIndex([a0 - pd.Timedelta(hours=5)])
        ev = evaluate_alarms(alarms, INCIDENTS["accident_1"],
                             (a0 - pd.Timedelta(days=30), a1 + pd.Timedelta(days=2)))
        self.assertTrue(ev["detected"])
        self.assertAlmostEqual(ev["lead_time_h"], 5.0)
        self.assertEqual(ev["fp"], 0)

    def test_false_alarm_outside_counts(self):
        a0, a1 = INCIDENTS["accident_1"]
        far = a0 - pd.Timedelta(days=20)
        ev = evaluate_alarms(pd.DatetimeIndex([far]), INCIDENTS["accident_1"],
                             (far - pd.Timedelta(days=10), a1))
        self.assertFalse(ev["detected"])
        self.assertEqual(ev["fp"], 1)

    def test_alarm_inside_window_not_lead(self):
        a0, a1 = INCIDENTS["accident_1"]
        alarms = pd.DatetimeIndex([a0 + pd.Timedelta(hours=10)])
        ev = evaluate_alarms(alarms, INCIDENTS["accident_1"],
                             (a0 - pd.Timedelta(days=1), a1 + pd.Timedelta(days=1)))
        self.assertTrue(ev["detected"])
        self.assertIsNone(ev["lead_time_h"])


class TestAlarmLogic(unittest.TestCase):
    def _sig(self, prob, r1, n=None):
        n = n or len(prob)
        idx = pd.date_range("2014-01-01", periods=n, freq="h")
        return pd.DataFrame({"prob_max": prob, "r1": r1, "y": 0.0, "med1": 0.0,
                             "q90": 1.0, "sigma": 1.0}, index=idx)

    def test_requires_conjunction(self):
        # только вероятность — нет тревоги (CUSUM молчит)
        sig = self._sig([0.9] * 10, [0.0] * 10)
        al = alarms_from_signals(sig, MonitorParams(p_star=0.3, persist_k=2, cooldown=2))
        self.assertEqual(len(al), 0)

    def test_all_conditions_fires(self):
        sig = self._sig([0.9] * 6, [2.0] * 6)
        al = alarms_from_signals(sig, MonitorParams(p_star=0.3, persist_k=2,
                                                    cusum_k=0.5, cusum_h=4.0, cooldown=24))
        self.assertGreaterEqual(len(al), 1)

    def test_cooldown_suppresses_burst(self):
        sig = self._sig([0.9] * 10, [2.0] * 10)
        prm = MonitorParams(p_star=0.3, persist_k=1, cooldown=24)
        al = alarms_from_signals(sig, prm)
        self.assertEqual(len(al), 1)


if __name__ == "__main__":
    unittest.main()
