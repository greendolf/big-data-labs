"""ЛР3: обработка одного микро-батча от извлечения до загрузки в raw.events.

Функции чистые от оркестратора — Prefect-задачи (prefect_flow.py) и юнит-тесты
вызывают их одинаково. Транзиентные сбои имитируются счётчиком попыток: extract
падает первые `flaky_failures` раз и успешается на следующей попытке — это сырьё
для демонстрации retries с экспоненциальной задержкой.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from lab3 import warehouse as wh
from lab3.contract import SchemaDriftError, apply_schema_contract, bad_rows_grouped_by_error, split_clean_and_bad

DEFAULT_ALLOWED_LATENESS_S = 4 * 3600  # окно «допустимого опоздания» событий


class TransientExtractError(RuntimeError):
    """Имитация транзиентного сбоя источника (сеть/диск) — побеждается retry'ом."""


_ATTEMPTS: dict[str, int] = {}


def extract(batch_file: Path | str, batch_id: str, flaky_failures: int = 0) -> pd.DataFrame:
    """Чтение батча; при flaky_failures=k первые k обращений к batch_id падают."""
    attempt = _ATTEMPTS.get(batch_id, 0) + 1
    _ATTEMPTS[batch_id] = attempt
    if attempt <= flaky_failures:
        raise TransientExtractError(
            f"{batch_id}: транзиентный сбой чтения (попытка {attempt}/{flaky_failures + 1})")
    return pd.read_parquet(batch_file)


def reset_attempts() -> None:
    _ATTEMPTS.clear()


def process_batch(batch_file: Path | str, batch_id: str, *, run_id: str | None = None,
                  warehouse: Path | None = None,
                  allowed_lateness_s: float = DEFAULT_ALLOWED_LATENESS_S,
                  flaky_failures: int = 0, alert_hook=None) -> dict[str, Any]:
    """Полный цикл батча. Возвращает словарь метрик; каждую стадию пишет в аудит."""
    run_id = run_id or wh.new_run_id()
    t_all = time.perf_counter()
    con = wh.connect(warehouse)
    try:
        # 1. extract (с имитацией транзиентных сбоев)
        t0 = time.perf_counter()
        df = extract(batch_file, batch_id, flaky_failures=flaky_failures)
        wh.audit(con, run_id, batch_id, "extract", "ok", rows_in=len(df),
                 duration_ms=int((time.perf_counter() - t0) * 1000))

        # 2. контракт схемы (schema drift: fallback на алиасы, алерт на незнакомцев)
        t0 = time.perf_counter()
        try:
            df, drift = apply_schema_contract(df)
        except SchemaDriftError as e:
            # Батч-уровневая ошибка: в DLQ уходит шапка батча, загрузка не выполняется.
            wh.dlq_add(con, "schema_contract", batch_id, run_id,
                       [{"batch_id": batch_id, "columns": sorted(pd.read_parquet(batch_file).columns)}],
                       "schema_drift_unrecoverable", str(e))
            wh.audit(con, run_id, batch_id, "schema_contract", "failed", metrics={"error": str(e)})
            raise
        if (drift["renamed_back"] or drift["dropped_unknown"]) and alert_hook:
            alert_hook(f"schema drift в {batch_id}: {drift}")
        wh.audit(con, run_id, batch_id, "schema_contract", "ok", duration_ms=int((time.perf_counter() - t0) * 1000),
                 metrics=drift)

        # 3. построчная валидация → DLQ
        t0 = time.perf_counter()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        clean, bad, tripped = split_clean_and_bad(df)
        dlq_total = 0
        for etype, rows in bad_rows_grouped_by_error(bad).items():
            dlq_total += wh.dlq_add(con, "row_validation", batch_id, run_id, rows, etype,
                                    f"batch {batch_id}")
        if dlq_total and alert_hook:
            alert_hook(f"DLQ: {dlq_total} строк из {batch_id} ({tripped})")
        wh.audit(con, run_id, batch_id, "row_validation", "ok", rows_in=len(df), rows_dlq=dlq_total,
                 duration_ms=int((time.perf_counter() - t0) * 1000),
                 metrics={"rules_tripped": tripped})

        # 4. загрузка: MERGE + watermark для late-arriving + снапшот
        t0 = time.perf_counter()
        wm = wh.watermark(con, allowed_lateness_s)
        metrics = wh.upsert_raw(con, clean, batch_id, run_id, watermark=wm)
        metrics["dlq"] = dlq_total
        snap = wh.snapshot_raw(con, run_id)
        metrics["snapshot"] = snap
        metrics["stage"] = "load_raw"
        metrics["batch_id"] = batch_id
        metrics["run_id"] = run_id
        metrics["duration_batch_ms"] = int((time.perf_counter() - t_all) * 1000)
        wh.audit(con, run_id, batch_id, "load_raw", "ok", rows_in=len(df),
                 rows_upserted=metrics["inserted"] + metrics["updated"],
                 rows_deduped=metrics["deduped_intra_batch"], rows_late=metrics["late"],
                 rows_dlq=dlq_total, duration_ms=metrics["duration_batch_ms"],
                 metrics=metrics)
        return metrics
    finally:
        con.close()
