"""Юнит-тесты процесса батча: транзиентные сбои (retries-сырьё) и полный цикл."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lab3 import warehouse as wh
from lab3.pipeline import TransientExtractError, process_batch, reset_attempts
from lab3.test_warehouse import make_batch


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="lab3_pipe_"))
        self.wh_path = self.dir / "warehouse.duckdb"
        wh.init_warehouse(self.wh_path)
        self.batch_file = self.dir / "b001.parquet"
        make_batch(25).to_parquet(self.batch_file, index=False)
        reset_attempts()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_flaky_extract_fails_then_succeeds(self):
        with self.assertRaises(TransientExtractError):
            process_batch(self.batch_file, "b001", warehouse=self.wh_path,
                          flaky_failures=1)
        m = process_batch(self.batch_file, "b001", warehouse=self.wh_path,
                          flaky_failures=1)  # вторая попытка успешна
        self.assertEqual(m["inserted"], 25)

    def test_full_cycle_metrics_and_audit(self):
        alerts: list[str] = []
        df = make_batch(20)
        df.loc[0, "label"] = 7  # одна грязная строка
        df.to_parquet(self.batch_file, index=False)
        m = process_batch(self.batch_file, "b001", warehouse=self.wh_path,
                          alert_hook=alerts.append)
        self.assertEqual(m["inserted"], 19)
        self.assertEqual(m["dlq"], 1)
        self.assertEqual(wh.dlq_count(wh.connect(self.wh_path)), 1)
        self.assertTrue(any("DLQ" in a for a in alerts))
        con = wh.connect(self.wh_path)
        stages = [r[0] for r in wh.run(
            con, "SELECT DISTINCT stage FROM ops.audit_runs ORDER BY stage")]
        con.close()
        self.assertEqual(stages, ["extract", "load_raw", "row_validation", "schema_contract"])

    def test_replay_is_noop(self):
        m1 = process_batch(self.batch_file, "b001", warehouse=self.wh_path)
        m2 = process_batch(self.batch_file, "b001", warehouse=self.wh_path)
        self.assertEqual((m2["inserted"], m2["updated"]), (0, m1["inserted"]))


if __name__ == "__main__":
    unittest.main()
