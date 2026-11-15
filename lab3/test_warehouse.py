"""Юнит-тесты хранилища: идемпотентный MERGE, снапшот/rollback, watermark, DLQ."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lab3 import warehouse as wh


def make_batch(n: int = 20, t0: str = "2026-11-01", devices: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    idx = range(n)
    df = pd.DataFrame({c: [float(i) for i in idx] for c in wh.FEATURE_COLUMNS})
    for c in wh.FEATURE_COLUMNS:  # признаки делаем неотрицательными и осмысленными
        df[c] = df[c].abs()
    df["Device_Name"] = [devices[i % len(devices)] for i in idx]
    df["Attack"] = ["Normal", "mirai"][0:1] * n
    df["Attack"] = ["Normal"] * n
    df["Attack_subType"] = ["none"] * n
    df["label"] = 1
    df["event_time"] = pd.Timestamp(t0) + pd.to_timedelta(list(idx), unit="s")
    return df


class WarehouseTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lab3_wh_"))
        self.wh_path = self.dir / "warehouse.duckdb"
        wh.init_warehouse(self.wh_path)
        self.con = wh.connect(self.wh_path)

    def tearDown(self):
        self.con.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_upsert_idempotent(self):
        df = make_batch(20)
        m1 = wh.upsert_raw(self.con, df, "b001", "run-1")
        self.assertEqual((m1["inserted"], m1["updated"]), (20, 0))
        m2 = wh.upsert_raw(self.con, df, "b001-replay", "run-2")
        self.assertEqual((m2["inserted"], m2["updated"]), (0, 20))
        self.assertEqual(wh.run(self.con, "SELECT count(*) FROM raw.events")[0][0], 20)

    def test_dedup_within_batch(self):
        df = pd.concat([make_batch(10), make_batch(10)], ignore_index=True)
        m = wh.upsert_raw(self.con, df, "b001", "run-1")
        self.assertEqual(m["inserted"], 10)
        self.assertEqual(m["deduped_intra_batch"], 10)

    def test_snapshot_and_rollback(self):
        wh.upsert_raw(self.con, make_batch(10), "b001", "run-1")
        snap = wh.snapshot_raw(self.con, "run-1")
        wh.upsert_raw(self.con, make_batch(30, t0="2026-11-02"), "b002", "run-2")
        self.assertEqual(wh.run(self.con, "SELECT count(*) FROM raw.events")[0][0], 40)
        rows = wh.rollback_raw(self.con, snap)
        self.assertEqual(rows, 10)
        self.assertEqual(wh.run(self.con, "SELECT count(*) FROM raw.events")[0][0], 10)

    def test_watermark_marks_late(self):
        df_new = make_batch(10, t0="2026-11-01 10:00")
        wh.upsert_raw(self.con, df_new, "b001", "run-1")
        wm = wh.watermark(self.con, 3600)  # допустимое опоздание 1 час
        self.assertIsNotNone(wm)
        df_late = make_batch(5, t0="2026-11-01 00:00")  # сильно раньше watermark
        m = wh.upsert_raw(self.con, df_late, "b002", "run-2", watermark=wm)
        self.assertEqual(m["late"], 5)
        self.assertEqual(
            wh.run(self.con, "SELECT count(*) FROM raw.events WHERE is_late")[0][0], 5)

    def test_dlq_roundtrip(self):
        n = wh.dlq_add(self.con, "row_validation", "b007", "run-1",
                       [{"label": 7, "_rule": "label_domain"}], "label_out_of_domain", "x")
        self.assertEqual(n, 1)
        self.assertEqual(wh.dlq_count(self.con), 1)
        row = wh.run(self.con, "SELECT row_json->>'$._rule' FROM ops.dlq")[0][0]
        self.assertEqual(row, "label_domain")


if __name__ == "__main__":
    unittest.main()
