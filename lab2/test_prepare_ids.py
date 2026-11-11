"""Unit tests for Lab 2 data preparation (dedup, slices, quality checks)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lab2.prepare_ids import create_prefix_slices, slice_report


HEADER_ROW_LIMIT_NOTE = "slices must cut on newline boundaries and stay under target bytes"
FIXTURE_MB = 100_000  # patched decimal MB so the small fixture can hold a 1 MB slice


def make_fixture(rows: int = 5_000) -> pd.DataFrame:
    base = pd.DataFrame({
        "MI_dir_L0.1_mean": [1.5, 2.25, -3.125, 4.0, 0.5] * (rows // 5),
        "HH_L0.1_magnitude": [10.0, 20.5, 30.25, 40.0, 50.75] * (rows // 5),
        "H_L0.1_variance": [0.0, 1.5, 2.25, 3.0, 4.75] * (rows // 5),
        "Device_Name": ["A_cam", "B_doorbell", "C_thermostat", "A_cam", "B_doorbell"] * (rows // 5),
        "Attack": ["mirai", "gafgyt", "Normal", "mirai", "gafgyt"] * (rows // 5),
        "Attack_subType": ["udp", "tcp", "Normal", "udp", "scan"] * (rows // 5),
        "label": [0, 0, 1, 0, 0] * (rows // 5),
    })
    return base


class PrepareIdsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.getenv("TEMP") or ".")
        self.root = Path(self.temp.name)
        self.source = self.root / "ids_cleaned.csv"
        frame = make_fixture()
        # Append exact duplicates so the source has repeated rows like the real data.
        pd.concat([frame, frame.head(500), frame.tail(500)]).to_csv(self.source, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_prefix_slices_are_row_aligned_and_within_target(self):
        with patch("lab2.prepare_ids.TARGET_BYTES_PER_MB", FIXTURE_MB):
            slices = create_prefix_slices(self.source, self.root / "slices", [1])
        path = slices[1]
        self.assertLessEqual(path.stat().st_size, FIXTURE_MB)
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            self.assertEqual(handle.read(1), b"\n", HEADER_ROW_LIMIT_NOTE)
        sliced = pd.read_csv(path)
        source = pd.read_csv(self.source)
        pd.testing.assert_frame_equal(
            sliced.reset_index(drop=True), source.iloc[: len(sliced)].reset_index(drop=True)
        )

    def test_slice_targets_must_be_smaller_than_source(self):
        with self.assertRaisesRegex(ValueError, "smaller than the source"):
            create_prefix_slices(self.source, self.root / "bad", [10_000])

    def test_invalid_size_arguments_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            create_prefix_slices(self.source, self.root / "bad", [0])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            create_prefix_slices(self.source, self.root / "bad", [1, 1])

    def test_slice_report_counts_rows_and_attack_share(self):
        with patch("lab2.prepare_ids.TARGET_BYTES_PER_MB", FIXTURE_MB):
            slices = create_prefix_slices(self.source, self.root / "slices2", [1])
        report = slice_report(slices)
        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["size_mb_target"], 1)
        self.assertEqual(
            sum(entry["attack_share"].values()), entry["rows"]
        )


if __name__ == "__main__":
    unittest.main()
