"""Unit tests for the Spark helpers.

Plan/stats/parser tests are pure Python; the session-dependent tests are
skipped when no working JDK can be resolved on the machine.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lab2.spark_pipeline import (
    plan_stats,
    skew_statistics,
    stage_summary,
    task_metrics,
)


EVENT_LOG_LINES = [
    {"Event": "SparkListenerTaskEnd", "Stage ID": 0,
     "Task Info": {"Task ID": 1, "Launch Time": 1000, "Finish Time": 2000},
     "Task Metrics": {"Input Metrics": {"Bytes Read": 500},
                      "Shuffle Write Metrics": {"Shuffle Bytes Written": 400}}},
    {"Event": "SparkListenerTaskEnd", "Stage ID": 0,
     "Task Info": {"Task ID": 2, "Launch Time": 1000, "Finish Time": 1500},
     "Task Metrics": {"Input Metrics": {"Bytes Read": 300},
                      "Shuffle Write Metrics": {"Shuffle Bytes Written": 100}}},
    {"Event": "SparkListenerTaskEnd", "Stage ID": 1,
     "Task Info": {"Task ID": 3, "Launch Time": 2100, "Finish Time": 9100},
     "Task Metrics": {"Shuffle Read Metrics": {"Remote Bytes Read": 9_000_000,
                                               "Local Bytes Read": 1_000_000},
                      "Shuffle Write Metrics": {"Shuffle Bytes Written": 0}}},
    {"Event": "SparkListenerTaskEnd", "Stage ID": 1,
     "Task Info": {"Task ID": 4, "Launch Time": 2100, "Finish Time": 2600},
     "Task Metrics": {"Shuffle Read Metrics": {"Remote Bytes Read": 100_000,
                                               "Local Bytes Read": 0}}},
    {"Event": "SparkListenerStageCompleted", "Stage Info": {"Stage ID": 0}},
]

PLAN_SAMPLE = """
== Physical Plan ==
*(2) HashAggregate(keys=[Attack#1, Attack_subType#2, Device_Name#3], functions=[...])
+- Exchange hashpartitioning(Attack#1, Attack_subType#2, Device_Name#3, 200), true, [id=#12]
   +- *(1) HashAggregate(keys=[...], functions=[partial_count, partial_avg(...)])
      +- FileScan csv default@default[...]
"""


class SparkPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.getenv("TEMP") or ".")
        self.event_dir = Path(self.temp.name)
        log = self.event_dir / "app-123"
        with log.open("w", encoding="utf-8") as handle:
            for line in EVENT_LOG_LINES:
                handle.write(json.dumps(line) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def test_task_metrics_parses_event_log(self):
        tasks = task_metrics(self.event_dir, "app-123")
        self.assertEqual(len(tasks), 4)
        self.assertEqual(int(tasks["shuffle_read_bytes"].sum()), 10_100_000)

    def test_task_metrics_respects_time_window(self):
        tasks = task_metrics(self.event_dir, "app-123", window_ms=(2050, 10_000))
        self.assertEqual(sorted(tasks["task_id"]), [3, 4])

    def test_stage_summary_reports_task_skew_ratio(self):
        tasks = task_metrics(self.event_dir, "app-123")
        summary = stage_summary(tasks)
        stage1 = summary[summary["stage_id"] == 1].iloc[0]
        self.assertEqual(int(stage1["tasks"]), 2)
        self.assertEqual(float(stage1["task_ms_max"]), 7000.0)
        self.assertGreater(float(stage1["task_time_skew_ratio"]), 1.0)
        self.assertGreater(float(stage1["shuffle_read_mb_max_task"]), 0.0)

    def test_skew_statistics_coefficient_of_variation(self):
        counts = pd.Series([50, 10, 10, 10, 10, 10])
        stats = skew_statistics(counts)
        mean = counts.mean()
        self.assertAlmostEqual(stats["coefficient_of_variation"],
                               round(counts.std(ddof=0) / mean, 2))
        self.assertAlmostEqual(stats["top1_share"], 0.5)
        self.assertEqual(stats["keys"], 6)

    def test_plan_stats_counts_exchange_and_aggregates(self):
        stats = plan_stats(PLAN_SAMPLE)
        self.assertEqual(stats["shuffle_exchanges"], 1)
        self.assertEqual(stats["hash_aggregates"], 2)
        self.assertGreaterEqual(stats["scans"], 1)


def _java_home_or_none() -> str | None:
    java_home = os.environ.get("JAVA_HOME")
    if java_home and (Path(java_home) / "bin" / "java.exe").is_file():
        return java_home
    root = Path(__file__).resolve().parents[1]
    fallback = root / "data" / "tools" / "lab2_java_home.txt"
    if fallback.is_file():
        candidate = fallback.read_text(encoding="utf-8").strip()
        if (Path(candidate) / "bin" / "java.exe").is_file():
            return candidate
    return None


class SparkSessionTests(unittest.TestCase):
    """End-to-end aggregation and salt parity on a small fixture with a real session."""

    @classmethod
    def setUpClass(cls):
        java_home = _java_home_or_none()
        if java_home is None:
            raise unittest.SkipTest("no JDK resolvable for PySpark")
        os.environ["JAVA_HOME"] = java_home
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
        hadoop_home = Path(__file__).resolve().parents[1] / "data" / "tools" / "hadoop"
        if (hadoop_home / "bin" / "winutils.exe").is_file():
            os.environ["HADOOP_HOME"] = str(hadoop_home)
        from lab2.spark_pipeline import build_session

        cls.temp = tempfile.TemporaryDirectory(dir=os.getenv("TEMP") or ".")
        cls.root = Path(cls.temp.name)
        cls.spark = build_session(cls.root / "events", name="lab2-tests", cores=1,
                                  shuffle_partitions=4)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "spark"):
            cls.spark.stop()
            cls.temp.cleanup()

    def _fixture(self) -> Path:
        rows = []
        for index in range(4000):
            hot = index % 3 == 0  # hot group like a botnet scanning pattern
            rows.append({
                "MI_dir_L0.1_mean": index % 17, "HH_L0.1_magnitude": index % 13,
                "H_L0.1_variance": (index % 7) * 1.5,
                "Device_Name": "hot_device" if hot else f"dev_{index % 9}",
                "Attack": "mirai" if hot else ["gafgyt", "Normal"][index % 2],
                "Attack_subType": "udp" if hot else ["tcp", "scan", "syn"][index % 3],
                "label": 0 if hot else index % 2,
            })
        path = self.root / "fixture.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_spark_aggregation_matches_pandas_and_salt_is_pure(self):
        from lab2.benchmark_engines import aggregate as cpu_aggregate
        from lab2.benchmark_engines import normalize
        from lab2.spark_pipeline import (
            aggregate_df, parity_frames, read_csv, result_pandas, salted_aggregate,
        )

        path = self._fixture()
        expected = normalize(cpu_aggregate("pandas", path))
        df = read_csv(self.spark, path)
        spark_result = result_pandas(aggregate_df(df))
        ok, reason = parity_frames(expected, spark_result)
        self.assertTrue(ok, reason)
        salted = result_pandas(salted_aggregate(df, buckets=5, seed=42))
        ok, reason = parity_frames(expected, salted)
        self.assertTrue(ok, f"salted parity broken: {reason}")


if __name__ == "__main__":
    unittest.main()
