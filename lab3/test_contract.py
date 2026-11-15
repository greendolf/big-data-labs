"""Юнит-тесты контракта схемы и построчной валидации (DLQ)."""
from __future__ import annotations

import unittest

import pandas as pd

from lab3.contract import SchemaDriftError, apply_schema_contract, split_clean_and_bad
from lab3.warehouse import FEATURE_COLUMNS


def frame(n: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({c: [1.0] * n for c in FEATURE_COLUMNS})
    df["event_time"] = pd.Timestamp("2026-11-01")
    df["Device_Name"] = "dev"
    df["Attack"] = "Normal"
    df["Attack_subType"] = "none"
    df["label"] = 1
    return df


class ContractTest(unittest.TestCase):
    def test_renamed_column_restored_via_alias(self):
        df = frame().rename(columns={"HH_L0.1_std": "HH_L0.1_SD"})
        out, report = apply_schema_contract(df)
        self.assertIn("HH_L0.1_std", out.columns)
        self.assertEqual(report["renamed_back"], {"HH_L0.1_SD": "HH_L0.1_std"})

    def test_unknown_column_dropped_with_report(self):
        df = frame()
        df["mystery_upstream"] = 1
        out, report = apply_schema_contract(df)
        self.assertNotIn("mystery_upstream", out.columns)
        self.assertEqual(report["dropped_unknown"], ["mystery_upstream"])

    def test_missing_required_raises_drift(self):
        df = frame().drop(columns=["Device_Name"])
        with self.assertRaises(SchemaDriftError):
            apply_schema_contract(df)

    def test_row_rules_split(self):
        df = frame(6)
        df.loc[0, "Device_Name"] = None        # null_key
        df.loc[1, "label"] = 7                  # label_out_of_domain
        df.loc[2, "H_L0.1_variance"] = -1.0     # negative_variance
        df.loc[3, "HH_L0.1_covariance"] = -999  # законное значение — НЕ в DLQ
        clean, bad, tripped = split_clean_and_bad(df)
        self.assertEqual(len(bad), 3)
        self.assertEqual(len(clean), 3)
        self.assertEqual(sorted(bad["_error_type"]),
                         ["label_out_of_domain", "negative_variance", "null_key"])
        self.assertNotIn(3, bad.index)  # отрицательная ковариация валидна
        self.assertEqual(len(tripped), 3)


if __name__ == "__main__":
    unittest.main()
