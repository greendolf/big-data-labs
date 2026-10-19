from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from analyze_ids import EXPECTED_COLUMNS, NUMERIC_COLUMNS, analyze_source


class AnalyzeIdsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "BoTNeTIoT-L01-v2.csv"
        rows = []
        attack = ["gafgyt", "Normal", "gafgyt"]
        subtype = ["combo", "Normal", "combo"]
        labels = ["0", "1", "0"]
        for row_index in range(3):
            row = {column: "1.25" for column in NUMERIC_COLUMNS}
            row.update({"Device_Name": "Device A", "Attack": attack[row_index], "Attack_subType": subtype[row_index], "label": labels[row_index]})
            if row_index == 1:
                row["HH_L0.1_pcc"] = "1.5"  # domain finding, but preserve data
            rows.append(row)
        rows[2] = rows[0].copy()  # exact full-row duplicate; retain it
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.runtime = self.root / "runtime"
        self.results = self.root / "results"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_value_pipeline_quality_and_roundtrip(self) -> None:
        profile = analyze_source(self.source, self.runtime, self.results)

        self.assertEqual(profile["source"]["rows"], 3)
        self.assertEqual(profile["source"]["columns"], 27)
        self.assertEqual(profile["schema"]["timestamp_columns"], [])
        self.assertEqual(profile["quality"]["label_category_consistency"]["invalid_rows"], 0)
        self.assertTrue(profile["quality"]["critical_label_gx"]["expectations_passed"])
        self.assertTrue(profile["quality"]["critical_label_gx"]["negative_control"]["failed_as_expected"])
        self.assertEqual(profile["quality"]["duplicates"]["full_row_count_excluding_first_occurrence"], 1)
        duplicate = profile["quality"]["duplicates"]["examples"][0]
        self.assertEqual(duplicate["source_rows"], [1, 3])
        self.assertGreater(profile["quality"]["numeric_domain_metrics"][5]["violations"], 0)
        self.assertEqual(profile["formats"]["csv"]["all_values_equal"], None)
        self.assertTrue(profile["formats"]["parquet"]["all_values_equal"])
        self.assertTrue(profile["formats"]["avro"]["all_values_equal"])
        self.assertTrue((self.runtime / "ids.parquet").is_file())
        self.assertTrue((self.runtime / "ids.avro").is_file())
        self.assertTrue((self.results / "category_counts.csv").is_file())

    def test_refuses_source_schema_drift(self) -> None:
        self.source.write_text("wrong,header\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected IDS schema"):
            analyze_source(self.source, self.runtime, self.results)


if __name__ == "__main__":
    unittest.main()
