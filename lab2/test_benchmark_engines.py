"""Unit tests for the cross-engine aggregation contract and resource metrics."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lab2.benchmark_engines import aggregate, normalize, parity


def make_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "MI_dir_L0.1_mean": [1.5, 2.25, -3.125, 4.0, 0.5, 7.75, -0.25, 3.5],
        "HH_L0.1_magnitude": [10.0, 20.5, 30.25, 40.0, 50.75, 60.0, 70.5, 80.25],
        "H_L0.1_variance": [0.0, 1.5, 2.25, 3.0, 4.75, 5.0, 6.5, 7.25],
        "Device_Name": ["A_cam", "B_doorbell", "C_thermostat", "A_cam",
                        "B_doorbell", "A_cam", "C_thermostat", "A_cam"],
        "Attack": ["mirai", "gafgyt", "Normal", "mirai",
                   "gafgyt", "mirai", "Normal", "mirai"],
        "Attack_subType": ["udp", "tcp", "Normal", "udp",
                           "scan", "udp", "Normal", "scan"],
        "label": [0, 0, 1, 0, 0, 0, 1, 0],
    })


class BenchmarkEnginesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.getenv("TEMP") or ".")
        self.source = Path(self.temp.name) / "fixture.csv"
        make_fixture().to_csv(self.source, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_pandas_aggregation_contract(self):
        result = normalize(aggregate("pandas", self.source))
        self.assertEqual(int(result["rows"].sum()), 8)
        self.assertEqual(
            set(zip(result["attack"], result["attack_subtype"], result["device"])),
            {("mirai", "udp", "A_cam"), ("mirai", "scan", "A_cam"),
             ("gafgyt", "tcp", "B_doorbell"), ("gafgyt", "scan", "B_doorbell"),
             ("Normal", "Normal", "C_thermostat")},
        )
        mirai_udp = result[(result.attack == "mirai") & (result.attack_subtype == "udp")].iloc[0]
        self.assertEqual(int(mirai_udp["rows"]), 3)
        self.assertAlmostEqual(float(mirai_udp["mean_MI_dir_mean"]), (1.5 + 4.0 + 7.75) / 3)
        self.assertAlmostEqual(float(mirai_udp["max_H_variance"]), 5.0)

    def test_engines_installed_match_pandas_result(self):
        expected = normalize(aggregate("pandas", self.source))
        for engine, module in (("dask", "dask"), ("polars", "polars"), ("duckdb", "duckdb")):
            with self.subTest(engine=engine):
                self.assertIsNotNone(importlib.util.find_spec(module), f"{module} must be installed")
                actual = normalize(aggregate(engine, self.source))
                ok, reason = parity(expected, actual)
                self.assertTrue(ok, reason)

    def test_parity_detects_perturbed_means(self):
        expected = normalize(aggregate("pandas", self.source))
        actual = expected.copy()
        actual.loc[0, "mean_MI_dir_mean"] += 1.0
        ok, reason = parity(expected, actual)
        self.assertFalse(ok)
        self.assertIn("mean_MI_dir_mean", reason)

    def test_parity_detects_row_count_drift(self):
        expected = normalize(aggregate("pandas", self.source))
        actual = expected.copy()
        actual.loc[0, "rows"] += 1
        ok, reason = parity(expected, actual)
        self.assertFalse(ok)

    def test_wide_read_includes_all_columns(self):
        result = normalize(aggregate("pandas", self.source, wide=True))
        self.assertEqual(int(result["rows"].sum()), 8)


if __name__ == "__main__":
    unittest.main()
