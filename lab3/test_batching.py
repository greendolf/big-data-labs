"""Юнит-тесты батчера: детерминизм, манифест, перенос поздних строк, фикстуры."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lab3.batching import build_batches, make_dirty, make_drift
from lab3.warehouse import FEATURE_COLUMNS


def source_frame(n: int = 400) -> pd.DataFrame:
    df = pd.DataFrame({c: [float(i % 7) for i in range(n)] for c in FEATURE_COLUMNS})
    df["Device_Name"] = [f"dev{i % 5}" for i in range(n)]
    df["Attack"] = ["Normal", "mirai", "gafgyt"][0] * n
    df["Attack_subType"] = ["none"] * n
    df["label"] = [i % 2 for i in range(n)]
    return df


class BatchingTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lab3_bat_"))
        self.src = self.dir / "src.parquet"
        source_frame(400).to_parquet(self.src, index=False)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_deterministic_and_manifest(self):
        m1 = build_batches(self.src, self.dir / "a", n_batches=4, late_share=0.0)
        m2 = build_batches(self.src, self.dir / "b", n_batches=4, late_share=0.0)
        self.assertEqual([b["rows"] for b in m1], [b["rows"] for b in m2])
        self.assertEqual(sum(b["rows"] for b in m1), 400)
        a = pd.read_parquet(self.dir / "a" / "b001.parquet")
        b = pd.read_parquet(self.dir / "b" / "b001.parquet")
        pd.testing.assert_frame_equal(a, b)

    def test_late_rows_moved_forward(self):
        import json
        out = self.dir / "c"
        m = build_batches(self.src, out, n_batches=6, late_share=0.1)
        self.assertEqual(sum(b["rows"] for b in m), 400)  # перенос ничего не теряет
        summary = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(summary["late_rows_moved"], 0)
        late_found = 0
        max_so_far = None  # стриминговый watermark: максимум event_time по всем прошлым батчам
        for b in m:
            df = pd.read_parquet(out / f"{b['batch_id']}.parquet")
            if max_so_far is not None:
                late_found += int((df["event_time"] < max_so_far).sum())
            batch_max = df["event_time"].max()
            max_so_far = batch_max if max_so_far is None else max(max_so_far, batch_max)
        self.assertGreater(late_found, summary["late_rows_moved"] // 2)

    def test_dirty_and_drift_fixtures(self):
        one = self.dir / "one.parquet"
        source_frame(50).to_parquet(one, index=False)
        stats = make_dirty(one, self.dir / "dirty.parquet", n_null_key=3,
                           n_bad_label=2, n_negative=2)
        df = pd.read_parquet(self.dir / "dirty.parquet")
        self.assertEqual(len(df), 50)
        self.assertEqual(int(df["Device_Name"].isna().sum()), 3)
        self.assertEqual(stats["total_injected"], 7)
        make_drift(one, self.dir / "drift.parquet")
        d = pd.read_parquet(self.dir / "drift.parquet")
        self.assertIn("HH_L0.1_SD", d.columns)
        self.assertNotIn("HH_L0.1_std", d.columns)
        self.assertIn("new_signal_from_upstream", d.columns)


if __name__ == "__main__":
    unittest.main()
