"""Lab 2 Parts 2-5: PySpark benchmark, physical plan analysis, skew, optimizations.

One local Spark session (local[2], 200 shuffle partitions as the assignment
prescribes) serves all Spark runs so that JVM start-up is excluded from every
measurement. Each timed action is bracketed by wall-clock and process-tree
resource sampling (Python driver plus its JVM child), and Spark event logging
is enabled so stage-level task metrics — task durations and shuffle bytes —
can be parsed back from the event log for plan and skew analysis.
"""

from __future__ import annotations

import contextlib
import json
import statistics  # noqa: F401  (used by the notebook for quick medians)
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from lab2.benchmark_engines import normalize

KEY_COLUMNS = ["Attack", "Attack_subType", "Device_Name"]
MEAN_COLUMNS = ["MI_dir_L0.1_mean", "HH_L0.1_magnitude"]
MAX_COLUMN = "H_L0.1_variance"


def dot_col(name: str):
    """Quote identifiers that contain dots for Spark 4's strict analyzer."""
    return F.col("`" + name + "`")


def build_session(event_log_dir: Path, name: str = "BigDataLab2", cores: int = 2,
                  shuffle_partitions: int = 200, extra: dict[str, str] | None = None) -> SparkSession:
    event_log_dir.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.appName(name)
        .master(f"local[{cores}]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", str(event_log_dir))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .config("spark.sql.session.timeZone", "UTC")
    )
    for key, value in (extra or {}).items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def aggregate_df(df: DataFrame) -> DataFrame:
    """The single aggregation contract used by every tool in this lab."""
    return df.groupBy(*KEY_COLUMNS).agg(
        F.count(F.lit(1)).alias("rows"),
        F.avg(dot_col(MEAN_COLUMNS[0])).alias("mean_MI_dir_mean"),
        F.avg(dot_col(MEAN_COLUMNS[1])).alias("mean_HH_magnitude"),
        F.max(dot_col(MAX_COLUMN)).alias("max_H_variance"),
    )


def read_csv(spark: SparkSession, path: Path) -> DataFrame:
    """Baseline CSV read with schema inference, matching a default notebook run."""
    return spark.read.csv(str(path), header=True, inferSchema=True)


def read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.parquet(str(path))


def result_pandas(result: DataFrame) -> pd.DataFrame:
    return normalize(result.toPandas())


def _tree_rss_bytes(root: psutil.Process) -> tuple[int, float, float, int, int]:
    total_rss = 0
    total_user = 0.0
    total_sys = 0.0
    total_read = 0
    total_write = 0
    procs = [root] + root.children(recursive=True)
    for proc in procs:
        try:
            total_rss += proc.memory_info().rss
            cpu = proc.cpu_times()
            total_user += cpu.user
            total_sys += cpu.system
            io = proc.io_counters()
            total_read += getattr(io, "read_bytes", 0)
            total_write += getattr(io, "write_bytes", 0)
        except (psutil.Error, AttributeError):
            continue
    return total_rss, total_user, total_sys, total_read, total_write


def timed(action: Callable[[], pd.DataFrame]) -> dict[str, Any]:
    """Time an action with process-tree (Python driver + JVM) resource sampling."""
    root = psutil.Process()
    peak_rss = [0]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.1):
            rss, _, _, _, _ = _tree_rss_bytes(root)
            peak_rss[0] = max(peak_rss[0], rss)

    _, cpu_before, sys_before, read_before, write_before = _tree_rss_bytes(root)
    started = time.perf_counter()
    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    t0_ms = int(time.time() * 1000)
    error = None
    frame: pd.DataFrame | None = None
    try:
        frame = action()
    except Exception as exc:  # noqa: BLE001 - benchmark harness records failures
        error = f"{type(exc).__name__}: {exc}"
    t1_ms = int(time.time() * 1000)
    stop.set()
    watcher.join(timeout=2)
    _, cpu_after, sys_after, read_after, write_after = _tree_rss_bytes(root)
    return {
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": max(0.0, (cpu_after + sys_after) - (cpu_before + sys_before)),
        "peak_rss_tree_mb": peak_rss[0] / 1_000_000,
        "read_bytes": max(0, read_after - read_before),
        "write_bytes": max(0, write_after - write_before),
        "window_start_ms": t0_ms,
        "window_end_ms": t1_ms,
        "error": error,
        "result": frame,
    }


def physical_plan(result: DataFrame, mode: str = "formatted") -> str:
    """Capture explain() output as a string (Spark prints it to stdout)."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result.explain(mode=mode)
    return buffer.getvalue()


def plan_stats(plan: str) -> dict[str, int]:
    return {
        "shuffle_exchanges": plan.count("Exchange"),
        "hash_aggregates": plan.count("HashAggregate"),
        "sorts": plan.count("Sort "),
        "scans": plan.count("Scan "),
    }


def _event_log_path(event_dir: Path, app_id: str) -> Path | None:
    for name in (app_id, f"in_progress_{app_id}"):
        candidate = event_dir / name
        if candidate.is_file():
            return candidate
    return None


def task_metrics(event_dir: Path, app_id: str, window_ms: tuple[int, int] | None = None) -> pd.DataFrame:
    """Parse TaskEnd metrics from the Spark event log, optionally within a time window."""
    path = _event_log_path(event_dir, app_id)
    if path is None:
        raise FileNotFoundError(f"No event log for application {app_id} in {event_dir}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if '"SparkListenerTaskEnd"' not in line:
                continue
            event = json.loads(line)
            metrics = event.get("Task Metrics") or {}
            info = event.get("Task Info") or {}
            finish = int(info.get("Finish Time", 0))
            if window_ms and not (window_ms[0] <= finish <= window_ms[1]):
                continue
            rows.append({
                "stage_id": int(event.get("Stage ID", -1)),
                "task_id": int(info.get("Task ID", -1)),
                "duration_ms": max(0, int(info.get("Finish Time", 0)) - int(info.get("Launch Time", 0))),
                "shuffle_read_bytes": int(metrics.get("Shuffle Read Metrics", {}).get("Remote Bytes Read", 0))
                + int(metrics.get("Shuffle Read Metrics", {}).get("Local Bytes Read", 0)),
                "shuffle_write_bytes": int(metrics.get("Shuffle Write Metrics", {}).get("Shuffle Bytes Written", 0)),
                "input_bytes": int(metrics.get("Input Metrics", {}).get("Bytes Read", 0)),
                "output_metrics_records": int(metrics.get("Output Metrics", {}).get("Records Written", 0)),
                "records_read": int(metrics.get("Input Metrics", {}).get("Records Read", 0)),
            })
    return pd.DataFrame(rows)


def stage_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    """Per-stage view: task counts, duration and shuffle skew, total shuffle bytes."""
    if tasks.empty:
        return pd.DataFrame()
    rows = []
    for stage_id, group in tasks.groupby("stage_id"):
        durations = group["duration_ms"]
        reads = group["shuffle_read_bytes"]
        rows.append({
            "stage_id": int(stage_id),
            "tasks": int(len(group)),
            "wall_ms": int(durations.max()),
            "task_ms_median": float(durations.median()),
            "task_ms_max": float(durations.max()),
            "task_time_skew_ratio": float(durations.max() / durations.median()) if durations.median() > 0 else float("inf"),
            "shuffle_read_mb_total": round(float(reads.sum()) / 1e6, 2),
            "shuffle_read_mb_median_task": round(float(reads.median()) / 1e6, 3),
            "shuffle_read_mb_max_task": round(float(reads.max()) / 1e6, 3),
            "shuffle_write_mb_total": round(float(group["shuffle_write_bytes"].sum()) / 1e6, 2),
            "input_mb_total": round(float(group["input_bytes"].sum()) / 1e6, 2),
        })
    return pd.DataFrame(rows).sort_values("stage_id").reset_index(drop=True)


def key_distribution(df: DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupBy(column).count()
        .orderBy(F.desc("count"))
        .toPandas()
    )


def skew_statistics(counts: pd.Series) -> dict[str, float]:
    """Coefficient of variation and top-key shares — the assignment's skew degree."""
    values = counts.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    ordered = values.sort_values(ascending=False)
    total = float(ordered.sum())
    return {
        "keys": int(len(values)),
        "rows": int(total),
        "mean_rows_per_key": round(mean, 1),
        "std_rows_per_key": round(std, 1),
        "coefficient_of_variation": round(std / mean, 2) if mean else float("inf"),
        "top1_share": round(float(ordered.iloc[0]) / total, 4) if len(ordered) else 0.0,
        "top10_share": round(float(ordered.head(10).sum()) / total, 4) if len(ordered) else 0.0,
        "max_over_median": round(float(ordered.iloc[0]) / float(ordered.median()), 1)
        if len(ordered) and ordered.median() > 0 else float("inf"),
    }


def salted_aggregate(df: DataFrame, buckets: int = 8, seed: int = 42) -> DataFrame:
    """Two-stage aggregation with salted keys to break hot groups apart."""
    salted = df.withColumn("_salt", (F.rand(seed) * F.lit(buckets)).cast("int"))
    partial = salted.groupBy(*KEY_COLUMNS, "_salt").agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(dot_col(MEAN_COLUMNS[0])).alias("sum_mi"),
        F.sum(dot_col(MEAN_COLUMNS[1])).alias("sum_hh"),
        F.max(dot_col(MAX_COLUMN)).alias("max_hv"),
    )
    return partial.groupBy(*KEY_COLUMNS).agg(
        F.sum("rows").alias("rows"),
        (F.sum("sum_mi") / F.sum("rows")).alias("mean_MI_dir_mean"),
        (F.sum("sum_hh") / F.sum("rows")).alias("mean_HH_magnitude"),
        F.max("max_hv").alias("max_H_variance"),
    )


def parity_frames(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str | None]:
    from lab2.benchmark_engines import parity

    return parity(expected, actual)


def repeated_run(action: Callable[[], pd.DataFrame], warmups: int = 1,
                 runs: int = 3, keep_result: bool = True) -> dict[str, Any]:
    """Warm up caches/JIT untimed, then take median and min/max of timed repeats."""
    for _ in range(warmups):
        action()
    timings: list[float] = []
    windows: list[tuple[int, int]] = []
    peak_rss = 0.0
    peak_cpu = 0.0
    peak_read = 0
    peak_write = 0
    error: str | None = None
    last_frame: pd.DataFrame | None = None
    for _ in range(runs):
        outcome = timed(action)
        timings.append(outcome["wall_seconds"])
        windows.append((outcome["window_start_ms"], outcome["window_end_ms"]))
        peak_rss = max(peak_rss, outcome["peak_rss_tree_mb"])
        peak_cpu = max(peak_cpu, outcome["cpu_seconds"])
        peak_read = max(peak_read, outcome["read_bytes"])
        peak_write = max(peak_write, outcome["write_bytes"])
        error = outcome["error"]
        if outcome["result"] is not None:
            last_frame = outcome["result"]
    return {
        "wall_seconds_median": float(np.median(timings)),
        "wall_seconds_min": float(np.min(timings)),
        "wall_seconds_max": float(np.max(timings)),
        "runs": runs,
        "windows": windows,
        "peak_rss_tree_mb": peak_rss,
        "cpu_seconds_peak": peak_cpu,
        "read_bytes_peak": peak_read,
        "write_bytes_peak": peak_write,
        "error": error,
        "result": last_frame if keep_result else None,
    }


@contextlib.contextmanager
def session_confs(spark: SparkSession, overrides: dict[str, str]):
    """Temporarily set Spark SQL configs and restore the previous values."""
    previous = {key: spark.conf.get(key, "") for key in overrides}
    for key, value in overrides.items():
        spark.conf.set(key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value:
                spark.conf.set(key, value)
            else:
                spark.conf.unset(key)


def _last_stage_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    """Keep only the stage ids of the final (largest) job window in the run."""
    if tasks.empty:
        return tasks
    return stage_summary(tasks)


def run_spark_sizes(spark: SparkSession, event_dir: Path, app_id: str,
                    inputs: dict[str, Path], results_dir: Path,
                    warmups: int = 1, runs: int = 3) -> pd.DataFrame:
    """Part 2 Spark legs: one repeated benchmark per input, uncached CSV reads."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in inputs.items():
        def action(path: Path = path) -> pd.DataFrame:
            return result_pandas(aggregate_df(read_csv(spark, path)))

        stats = repeated_run(action, warmups=warmups, runs=runs)
        window = stats["windows"][-1] if stats["windows"] else None
        tasks = task_metrics(event_dir, app_id, window)
        stages = _last_stage_summary(tasks)
        stages_path = results_dir / f"spark_stages_{label}.csv"
        stages.to_csv(stages_path, index=False)
        if stats["result"] is not None:
            stats["result"].to_csv(results_dir / f"spark_result_{label}.csv", index=False)
        rows.append({
            "engine": "pyspark",
            "size_label": label,
            "input_bytes": path.stat().st_size,
            "wall_seconds": stats["wall_seconds_median"],
            "wall_seconds_min": stats["wall_seconds_min"],
            "wall_seconds_max": stats["wall_seconds_max"],
            "cpu_seconds": stats["cpu_seconds_peak"],
            "peak_rss_mb": stats["peak_rss_tree_mb"],
            "read_bytes": stats["read_bytes_peak"],
            "write_bytes": stats["write_bytes_peak"],
            "runs": runs,
            "error": stats["error"],
            "stages_csv": str(stages_path),
        })
    frame = pd.DataFrame(rows)
    return frame


def analyze_physical_plan(spark: SparkSession, event_dir: Path, app_id: str,
                          csv_path: Path, results_dir: Path) -> dict[str, Any]:
    """Part 3: dump explain() plans and the stage/task profile of one real run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv(spark, csv_path)
    result = aggregate_df(df)
    plan_formatted = physical_plan(result, mode="formatted")
    plan_extended = physical_plan(result, mode="extended")
    (results_dir / "spark_physical_plan.txt").write_text(
        "=== explain(formatted) ===\n" + plan_formatted
        + "\n\n=== explain(extended) ===\n" + plan_extended, encoding="utf-8"
    )
    stats = repeated_run(lambda: result_pandas(result), warmups=0, runs=1)
    window = stats["windows"][-1] if stats["windows"] else None
    tasks = task_metrics(event_dir, app_id, window)
    stages = stage_summary(tasks)
    stages.to_csv(results_dir / "spark_plan_stage_summary.csv", index=False)
    tasks.to_csv(results_dir / "spark_plan_task_metrics.csv", index=False)
    return {
        "plan_formatted": plan_formatted,
        "plan_stats": plan_stats(plan_formatted),
        "wall_seconds": stats["wall_seconds_median"],
        "stages": stages,
        "tasks": tasks,
    }


def skew_experiment(spark: SparkSession, event_dir: Path, app_id: str,
                    parsed_df: DataFrame, results_dir: Path, buckets: int = 8,
                    runs: int = 3) -> dict[str, Any]:
    """Part 4: key-distribution stats plus baseline vs salted timing and plans."""
    results_dir.mkdir(parents=True, exist_ok=True)
    distributions: dict[str, pd.DataFrame] = {}
    statistics_by_column: dict[str, dict[str, float]] = {}
    for column in KEY_COLUMNS:
        distribution = key_distribution(parsed_df, column)
        distribution.to_csv(results_dir / f"key_distribution_{column}.csv", index=False)
        distributions[column] = distribution
        statistics_by_column[column] = skew_statistics(distribution["count"])
    composite = (
        parsed_df.groupBy(*KEY_COLUMNS).count().orderBy(F.desc("count")).toPandas()
    )
    composite.to_csv(results_dir / "key_distribution_composite.csv", index=False)
    statistics_by_column["composite_key"] = skew_statistics(composite["count"])

    baseline_stats = repeated_run(lambda: result_pandas(aggregate_df(parsed_df)), runs=runs)
    salted_stats = repeated_run(
        lambda: result_pandas(salted_aggregate(parsed_df, buckets=buckets)), runs=runs
    )
    baseline_plan = physical_plan(aggregate_df(parsed_df))
    salted_plan = physical_plan(salted_aggregate(parsed_df, buckets=buckets))
    (results_dir / "skew_baseline_plan.txt").write_text(baseline_plan, encoding="utf-8")
    (results_dir / "skew_salted_plan.txt").write_text(salted_plan, encoding="utf-8")

    parity_ok, parity_error = (
        parity_frames(baseline_stats["result"], salted_stats["result"])
        if baseline_stats["result"] is not None and salted_stats["result"] is not None
        else (False, "a run failed")
    )
    baseline_window = baseline_stats["windows"][-1]
    salted_window = salted_stats["windows"][-1]
    baseline_stages = stage_summary(task_metrics(event_dir, app_id, baseline_window))
    salted_stages = stage_summary(task_metrics(event_dir, app_id, salted_window))
    baseline_stages.assign(variant="baseline").to_csv(results_dir / "skew_baseline_stages.csv", index=False)
    salted_stages.assign(variant=f"salted_{buckets}").to_csv(results_dir / "skew_salted_stages.csv", index=False)
    return {
        "distributions": distributions,
        "statistics": statistics_by_column,
        "composite": composite,
        "buckets": buckets,
        "baseline": baseline_stats,
        "salted": salted_stats,
        "baseline_stages": baseline_stages,
        "salted_stages": salted_stages,
        "baseline_plan": baseline_plan,
        "salted_plan": salted_plan,
        "plan_stats_baseline": plan_stats(baseline_plan),
        "plan_stats_salted": plan_stats(salted_plan),
        "parity_ok": parity_ok,
        "parity_error": parity_error,
    }


def optimization_matrix(spark: SparkSession, csv_path: Path, parquet_path: Path,
                        cached_df: DataFrame, event_dir: Path, results_dir: Path,
                        partition_sweep: tuple[int, ...] = (1, 4, 16, 64, 400),
                        runs: int = 3) -> dict[str, Any]:
    """Part 5: measure input-format and query-level optimizations separately.

    Group A compares uncached end-to-end runs (CSV vs Parquet vs Parquet + column
    pruning). Group B compares config/query variants on one cached parsed CSV
    frame so that each row isolates exactly one change; the reference for each
    row is its own group baseline. This split is disclosed in the report.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    def action_csv() -> pd.DataFrame:
        return result_pandas(aggregate_df(read_csv(spark, csv_path)))

    def action_parquet() -> pd.DataFrame:
        return result_pandas(aggregate_df(read_parquet(spark, parquet_path)))

    def action_parquet_pruned() -> pd.DataFrame:
        df = read_parquet(spark, parquet_path).select(
            *[F.col(name) for name in KEY_COLUMNS],
            dot_col(MEAN_COLUMNS[0]), dot_col(MEAN_COLUMNS[1]), dot_col(MAX_COLUMN),
        )
        return result_pandas(aggregate_df(df))

    group_a_rows: list[dict[str, Any]] = []
    base_a = repeated_run(action_csv, runs=runs, keep_result=False)
    group_a_rows.append({
        "id": "a_baseline_csv", "group": "A (uncached end-to-end)",
        "description": "baseline: CSV read (schema inference) + aggregation",
        **{k: base_a[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max",
                                  "peak_rss_tree_mb", "error")},
        "effort": 0, "reference": "a_baseline_csv",
    })
    for spec in (
        ("a_parquet_input", "Parquet (zstd) instead of CSV", action_parquet, 2),
        ("a_parquet_pruned", "Parquet + explicit column pruning before aggregation",
         action_parquet_pruned, 3),
    ):
        variant_id, description, action, effort = spec
        stats = repeated_run(action, runs=runs, keep_result=False)
        group_a_rows.append({
            "id": variant_id, "group": "A (uncached end-to-end)", "description": description,
            **{k: stats[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max",
                                     "peak_rss_tree_mb", "error")},
            "effort": effort, "reference": "a_baseline_csv",
        })

    def action_cached() -> pd.DataFrame:
        return result_pandas(aggregate_df(cached_df))

    group_b_rows: list[dict[str, Any]] = []
    base_b = repeated_run(action_cached, warmups=0, runs=runs, keep_result=False)
    group_b_rows.append({
        "id": "b_baseline_cached", "group": "B (cached parsed CSV)",
        "description": "baseline: cached parsed CSV, AQE on, 200 partitions",
        **{k: base_b[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max",
                                  "peak_rss_tree_mb", "error")},
        "effort": 0, "reference": "b_baseline_cached",
    })
    with session_confs(spark, {"spark.sql.adaptive.enabled": "false"}):
        stats = repeated_run(action_cached, warmups=0, runs=runs, keep_result=False)
    group_b_rows.append({
        "id": "b_aqe_off", "group": "B (cached parsed CSV)",
        "description": "AQE disabled (regression check, negative optimization)",
        **{k: stats[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max",
                                 "peak_rss_tree_mb", "error")},
        "effort": 0, "reference": "b_baseline_cached",
    })
    sweep: list[dict[str, Any]] = []
    for partitions in partition_sweep:
        with session_confs(spark, {"spark.sql.shuffle.partitions": str(partitions),
                                   "spark.default.parallelism": str(partitions)}):
            stats = repeated_run(action_cached, warmups=0, runs=runs, keep_result=False)
        sweep.append({
            "id": f"b_partitions_{partitions}", "shuffle_partitions": partitions,
            "wall_seconds_median": stats["wall_seconds_median"],
            "wall_seconds_min": stats["wall_seconds_min"],
            "wall_seconds_max": stats["wall_seconds_max"],
            "error": stats["error"],
        })
    best = min(sweep, key=lambda row: row["wall_seconds_median"])
    best_partitions = int(best["shuffle_partitions"])
    group_b_rows.append({
        "id": f"b_partitions_{best_partitions}", "group": "B (cached parsed CSV)",
        "description": f"best shuffle.partitions from sweep {list(partition_sweep)} "
                       f"(best over baseline 200)",
        **{k: best[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max")},
        "peak_rss_tree_mb": None, "error": best["error"],
        "effort": 1, "reference": "b_baseline_cached",
    })

    def action_repartition() -> pd.DataFrame:
        df = cached_df.repartition(best_partitions, *KEY_COLUMNS)
        return result_pandas(aggregate_df(df))

    stats = repeated_run(action_repartition, warmups=0, runs=runs, keep_result=False)
    group_b_rows.append({
        "id": "b_repartition_keys", "group": "B (cached parsed CSV)",
        "description": f"repartition({best_partitions}, keys) before groupBy",
        **{k: stats[k] for k in ("wall_seconds_median", "wall_seconds_min", "wall_seconds_max",
                                 "peak_rss_tree_mb", "error")},
        "effort": 2, "reference": "b_baseline_cached",
    })

    iterations = []
    df_iter = read_csv(spark, csv_path).cache()
    try:
        first = timed(lambda: result_pandas(aggregate_df(df_iter)))
        second = timed(lambda: result_pandas(aggregate_df(df_iter)))
        iterations = [first["wall_seconds"], second["wall_seconds"]]
    finally:
        df_iter.unpersist()
    group_b_rows.append({
        "id": "b_cache_iteration", "group": "B (cached parsed CSV)",
        "description": "persist() of the parsed CSV: 2nd query on cached input",
        "wall_seconds_median": iterations[1], "wall_seconds_min": iterations[1],
        "wall_seconds_max": iterations[1], "peak_rss_tree_mb": second["peak_rss_tree_mb"],
        "error": first["error"] or second["error"],
        "effort": 1, "reference": "a_baseline_csv",
        "first_query_with_cache_seconds": iterations[0],
    })

    with session_confs(spark, {"spark.sql.shuffle.partitions": str(best_partitions),
                               "spark.default.parallelism": str(best_partitions)}):
        combo_stats = repeated_run(action_parquet_pruned, runs=runs, keep_result=False)
    combo_row = {
        "id": "combination", "group": "C (best combination)",
        "description": f"Parquet + pruning + AQE on + partitions={best_partitions}",
        **{k: combo_stats[k] for k in ("wall_seconds_median", "wall_seconds_min",
                                       "wall_seconds_max", "peak_rss_tree_mb", "error")},
        "effort": 3, "reference": "a_baseline_csv",
    }

    measurements = pd.DataFrame(group_a_rows + group_b_rows + [combo_row])
    baseline_times = {row["id"]: row["wall_seconds_median"]
                      for row in group_a_rows + group_b_rows}
    measurements["reference_seconds"] = measurements["reference"].map(baseline_times)
    measurements["speedup_vs_reference"] = (
        measurements["reference_seconds"] / measurements["wall_seconds_median"]
    )
    measurements["speedup_vs_csv_baseline"] = (
        baseline_times["a_baseline_csv"] / measurements["wall_seconds_median"]
    )
    measurements.to_csv(results_dir / "optimization_measurements.csv", index=False)
    pd.DataFrame(sweep).to_csv(results_dir / "partition_sweep.csv", index=False)
    return {
        "measurements": measurements,
        "sweep": pd.DataFrame(sweep),
        "best_partitions": best_partitions,
        "baseline_a_seconds": baseline_times["a_baseline_csv"],
        "baseline_b_seconds": baseline_times["b_baseline_cached"],
        "combination": combo_row,
    }
