"""Lab 2 Parts 2 and 6: CPU engine benchmarks for the shared aggregation task.

Every engine runs in its own subprocess on identical cleaned CSV inputs, so a
crash or memory blow-up of one engine cannot contaminate the others. The
worker samples peak RSS every 50 ms and records wall time, CPU time, and
Windows I/O counters around exactly the same contract: read the CSV columns,
aggregate count/mean/mean/max by Attack x Attack_subType x Device_Name, and
write the result as CSV. Engine imports and interpreter start-up are excluded.

Thread limits (POLARS_MAX_THREADS and friends) are inherited from the parent,
which pins every CPU engine to the same two-thread budget. Results are checked
against the successful Pandas baseline on the same input before being trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

KEY_COLUMNS = ["Attack", "Attack_subType", "Device_Name"]
MEAN_COLUMNS = ["MI_dir_L0.1_mean", "HH_L0.1_magnitude"]
MAX_COLUMN = "H_L0.1_variance"
RESULT_COLUMNS = [
    "attack", "attack_subtype", "device", "rows",
    "mean_MI_dir_mean", "mean_HH_magnitude", "max_H_variance",
]

WORKER_THREADS = 2
SLICE_TIMEOUT_SECONDS = 15 * 60
FULL_TIMEOUT_SECONDS = 30 * 60
CPU_ENGINES = ["pandas", "dask", "polars", "duckdb"]


def aggregate(engine: str, csv_path: Path, wide: bool = False) -> pd.DataFrame:
    """Execute the identical aggregation contract with the requested engine."""
    if engine == "pandas":
        if wide:
            # Honest worst case: load every column with default dtypes, exactly
            # what a naive notebook run would do with a new dataset.
            frame = pd.read_csv(csv_path)
        else:
            frame = pd.read_csv(
                csv_path, usecols=KEY_COLUMNS + MEAN_COLUMNS + [MAX_COLUMN],
                dtype={column: "string" for column in KEY_COLUMNS},
            )
        result = frame.groupby(KEY_COLUMNS, dropna=False).agg(
            rows=("Attack", "size"),
            mean_MI_dir_mean=("MI_dir_L0.1_mean", "mean"),
            mean_HH_magnitude=("HH_L0.1_magnitude", "mean"),
            max_H_variance=("H_L0.1_variance", "max"),
        ).reset_index()
        return result

    if engine == "dask":
        import dask.dataframe as dd

        frame = dd.read_csv(
            str(csv_path), usecols=KEY_COLUMNS + MEAN_COLUMNS + [MAX_COLUMN],
            blocksize="64MB", dtype={column: "object" for column in KEY_COLUMNS},
        )
        result = frame.groupby(KEY_COLUMNS, dropna=False).agg(
            rows=("Attack", "size"),
            mean_MI_dir_mean=("MI_dir_L0.1_mean", "mean"),
            mean_HH_magnitude=("HH_L0.1_magnitude", "mean"),
            max_H_variance=("H_L0.1_variance", "max"),
        ).compute(scheduler="threads", num_workers=WORKER_THREADS).reset_index()
        return result

    if engine == "polars":
        import polars as pl

        return (
            pl.scan_csv(
                csv_path,
                schema_overrides={**{column: pl.String for column in KEY_COLUMNS},
                                  **{column: pl.Float64 for column in MEAN_COLUMNS + [MAX_COLUMN]}},
            )
            .select(KEY_COLUMNS + MEAN_COLUMNS + [MAX_COLUMN])
            .group_by(KEY_COLUMNS)
            .agg(
                pl.len().alias("rows"),
                pl.col("MI_dir_L0.1_mean").mean().alias("mean_MI_dir_mean"),
                pl.col("HH_L0.1_magnitude").mean().alias("mean_HH_magnitude"),
                pl.col("H_L0.1_variance").max().alias("max_H_variance"),
            )
            .collect()
            .to_pandas()
        )

    if engine == "duckdb":
        import duckdb

        path_sql = str(csv_path.resolve()).replace("'", "''")
        duckdb.sql("SET threads = 2")
        return duckdb.sql(
            f"""
            SELECT "Attack" AS attack, "Attack_subType" AS attack_subtype, "Device_Name" AS device,
                   COUNT(*) AS rows,
                   AVG(CAST("MI_dir_L0.1_mean" AS DOUBLE)) AS mean_MI_dir_mean,
                   AVG(CAST("HH_L0.1_magnitude" AS DOUBLE)) AS mean_HH_magnitude,
                   MAX(CAST("H_L0.1_variance" AS DOUBLE)) AS max_H_variance
            FROM read_csv('{path_sql}', header=true, all_varchar=true)
            GROUP BY 1, 2, 3
            """
        ).df()

    raise ValueError(f"Unknown engine: {engine}")


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={
        "Attack": "attack", "Attack_subType": "attack_subtype", "Device_Name": "device",
    }).copy()
    for column in ("attack", "attack_subtype", "device"):
        result[column] = result[column].astype(str)
    result["rows"] = result["rows"].astype("int64")
    for column in ("mean_MI_dir_mean", "mean_HH_magnitude", "max_H_variance"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[RESULT_COLUMNS].sort_values(
        ["attack", "attack_subtype", "device"]
    ).reset_index(drop=True)


def parity(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str | None]:
    keys = ["attack", "attack_subtype", "device", "rows"]
    if not expected[keys].equals(actual[keys]):
        return False, "group keys or row counts differ from the Pandas baseline"
    for column in ("mean_MI_dir_mean", "mean_HH_magnitude"):
        if not np.allclose(expected[column], actual[column], rtol=1e-5, atol=1e-5, equal_nan=True):
            return False, f"{column} differs from the Pandas baseline"
    if not np.allclose(expected["max_H_variance"], actual["max_H_variance"], rtol=1e-9, atol=1e-9):
        return False, "max_H_variance differs from the Pandas baseline"
    return True, None


def measure(worker_args: argparse.Namespace) -> dict[str, Any]:
    # Warm up engine imports outside the timed section so all engines are
    # timed on the same scope: CSV read, aggregation, and result serialization.
    if worker_args.engine == "dask":
        import dask.dataframe  # noqa: F401
    elif worker_args.engine == "polars":
        import polars  # noqa: F401
    elif worker_args.engine == "duckdb":
        import duckdb  # noqa: F401

    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = [start_rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.05):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    io_before = process.io_counters()
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    watcher = threading.Thread(target=sample_rss, daemon=True)
    watcher.start()
    error = None
    try:
        result = normalize(aggregate(worker_args.engine, Path(worker_args.input),
                                     wide=worker_args.wide))
        result.to_csv(worker_args.result, index=False)
    except Exception as exc:  # noqa: BLE001 - the worker reports any failure honestly
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop.set()
        watcher.join(timeout=1)
    cpu_after = process.cpu_times()
    io_after = process.io_counters()

    def io_attr(counters, *names) -> int:
        for name in names:
            value = getattr(counters, name, None)
            if value is not None:
                return int(value)
        return -1

    return {
        "engine": worker_args.engine,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
        "peak_rss_mb": peak_rss[0] / 1_000_000,
        "rss_delta_mb": max(0, peak_rss[0] - start_rss) / 1_000_000,
        "read_bytes": max(0, io_after.read_bytes - io_before.read_bytes),
        "write_bytes": max(0, io_after.write_bytes - io_before.write_bytes),
        "read_operations": max(0, io_attr(io_after, "read_operation_count", "ReadOperationCount")
                               - io_attr(io_before, "read_operation_count", "ReadOperationCount")),
        "write_operations": max(0, io_attr(io_after, "write_operation_count", "WriteOperationCount")
                                - io_attr(io_before, "write_operation_count", "WriteOperationCount")),
        "error": error,
    }


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(WORKER_THREADS),
        "OPENBLAS_NUM_THREADS": str(WORKER_THREADS),
        "MKL_NUM_THREADS": str(WORKER_THREADS),
        "NUMEXPR_NUM_THREADS": str(WORKER_THREADS),
        "POLARS_MAX_THREADS": str(WORKER_THREADS),
    })
    return env


def _launch(engine: str, input_path: Path, result_path: Path, wide: bool,
            timeout: int) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--engine", engine, "--input", str(input_path), "--result", str(result_path),
    ]
    if wide:
        command.append("--wide")
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=_worker_env(),
        )
    except subprocess.TimeoutExpired:
        return {"engine": engine, "error": f"worker timed out after {timeout} seconds"}
    except OSError as exc:
        return {"engine": engine, "error": f"could not start worker: {type(exc).__name__}: {exc}"}
    payload = next(
        (line for line in reversed(process.stdout.splitlines())
         if line.startswith("WORKER_RESULT=")), None
    )
    if payload is None:
        return {"engine": engine,
                "error": process.stderr[-1500:] or process.stdout[-1500:] or f"worker exited {process.returncode}"}
    metrics = json.loads(payload.split("=", 1)[1])
    if process.returncode != 0 and not metrics.get("error"):
        metrics["error"] = f"worker exited with status {process.returncode}"
    return metrics


def run_benchmark(slice_paths: dict[str, Path], output_dir: Path,
                  engines: list[str] = CPU_ENGINES) -> pd.DataFrame:
    """Run every engine on every input with parity against the Pandas baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for size_label, input_path in slice_paths.items():
        baseline_path: Path | None = None
        for engine in engines:
            result_path = output_dir / f"{size_label}_{engine}_result.csv"
            timeout = FULL_TIMEOUT_SECONDS if size_label == "full" else SLICE_TIMEOUT_SECONDS
            metrics = _launch(engine, input_path, result_path, wide=False, timeout=timeout)
            metrics.update({"size_label": size_label, "input_bytes": input_path.stat().st_size})
            metrics["parity_with_pandas"] = False
            if not metrics.get("error"):
                if engine == "pandas":
                    baseline_path = result_path
                    metrics["parity_with_pandas"] = True
                elif baseline_path is not None:
                    try:
                        ok, reason = parity(
                            normalize(pd.read_csv(baseline_path)),
                            normalize(pd.read_csv(result_path)),
                        )
                        metrics["parity_with_pandas"] = ok
                        if not ok:
                            metrics["parity_error"] = reason
                    except Exception as exc:  # noqa: BLE001
                        metrics["parity_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    metrics["parity_error"] = "successful Pandas baseline is unavailable"
            rows.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "engines_measurements.csv", index=False)
    return results


def run_full_pandas_check(full_csv: Path, output_dir: Path) -> dict[str, Any]:
    """Assignment check: can Pandas handle the full dataset at all (wide read)?"""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "full_pandas_wide_result.csv"
    metrics = _launch("pandas", full_csv, result_path, wide=True, timeout=FULL_TIMEOUT_SECONDS)
    metrics["input_bytes"] = full_csv.stat().st_size
    if metrics.get("error"):
        verdict = f"## OK ## Pandas не смог обработать полный датасет: {metrics['error']}"
    else:
        verdict = (
            "## WARNING ## Pandas обработал полный датасет — машина объёмом 32 ГБ не является "
            "препятствием; см. анализ пиковой памяти и экстраполяцию в разделе 2"
        )
    metrics["verdict"] = verdict
    (output_dir / "full_pandas_check.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--engine", choices=CPU_ENGINES)
    parser.add_argument("--input", type=str)
    parser.add_argument("--result", type=str)
    parser.add_argument("--wide", action="store_true", help="read every column with default dtypes")
    args = parser.parse_args()
    if args.worker:
        print("WORKER_RESULT=" + json.dumps(measure(args), ensure_ascii=False))
        return
    parser.error("this module is driven from the notebook or its tests")


if __name__ == "__main__":
    main()
