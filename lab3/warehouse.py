"""ЛР3: хранилище пайплайна Fraud Detection на DuckDB.

Схемы:
  raw       — реестр событий-«операций» (MERGE по idempotency_key)
  ops       — dlq, аудит прогонов, метаданные снапшотов
  snapshots — полные копии raw.events на момент коммита батча (Time Travel/rollback)

DuckDB — один писатель на файл: все функции открывают КОРОТКОЖИВУЮ сессию
(открыл → сделал → закрыл), чтобы dbt/Prefect-подпроцессы не ловили lock.
Пути строятся от ROOT (ожидается запуск из корня репозитория).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

ROOT = Path.cwd()
WORKDIR = ROOT / "data" / "lab3"
WAREHOUSE = WORKDIR / "warehouse.duckdb"

FEATURE_COLUMNS = [
    "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std", "HH_L0.1_magnitude",
    "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
    "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std", "HpHp_L0.1_magnitude",
    "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc",
]
KEY_COLUMNS = ["Device_Name", "Attack", "Attack_subType", "label"]
# Полный контракт «сырого» события: 27 колонок датасета + служебные event_time/ключ.
RAW_COLUMNS = FEATURE_COLUMNS + ["Device_Name", "Attack", "Attack_subType", "label"]

# Инвариант «>= 0» проверят только столбцы, которые действительно неотрицательны
# в источнике (проверено min() по всем 6.4М строк). covariance намеренно ИСКЛЮЧЁН:
# он законно принимает отрицательные значения (min ≈ −2.6e5, см. ЛР2).
NONNEGATIVE_COLUMNS = [c for c in FEATURE_COLUMNS
                       if c.endswith(("_variance", "_std", "_weight", "_magnitude", "_radius"))]
LABEL_SET = [0, 1]
ATTACK_DOMAIN = ["Normal", "mirai", "gafgyt"]


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def run(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None):
    """execute + fetchall: в duckdb 2.x нельзя мешать execute без доедания результата."""
    return con.execute(sql, params or []).fetchall()


def connect(warehouse: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = warehouse or WAREHOUSE
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def make_idempotency_key(df: pd.DataFrame) -> pd.Series:
    """md5 от полного кортежа: стабилен между повторными загрузками одних и тех
    же строк (защита от дублей при re-run) и уникален для разных событий."""
    import hashlib

    def one(row: tuple) -> str:
        parts = []
        for v in row:
            if isinstance(v, float):
                parts.append(repr(round(v, 12)))
            else:
                parts.append("" if v is None or (isinstance(v, float) and v != v) else str(v))
        return hashlib.md5("\x1f".join(parts).encode("utf-8")).hexdigest()

    cols = RAW_COLUMNS + ["event_time"]
    return pd.Series([one(tuple(r)) for r in df[cols].itertuples(index=False, name=None)],
                     index=df.index, name="idempotency_key")


def init_warehouse(warehouse: Path | None = None) -> None:
    con = connect(warehouse)
    try:
        run(con, "CREATE SCHEMA IF NOT EXISTS raw")
        run(con, "CREATE SCHEMA IF NOT EXISTS ops")
        run(con, "CREATE SCHEMA IF NOT EXISTS snapshots")
        feats = ",\n                ".join(f"{quote(c)} DOUBLE" for c in FEATURE_COLUMNS)
        run(con, f"""
            CREATE TABLE IF NOT EXISTS raw.events (
                idempotency_key VARCHAR PRIMARY KEY,
                batch_id        VARCHAR NOT NULL,
                event_time      TIMESTAMP NOT NULL,
                ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                is_late         BOOLEAN NOT NULL DEFAULT FALSE,
                {feats},
                "Device_Name"    VARCHAR,
                "Attack"         VARCHAR,
                "Attack_subType" VARCHAR,
                label            BIGINT
            )""")
        run(con, """
            CREATE TABLE IF NOT EXISTS ops.dlq (
                dlq_id     BIGINT PRIMARY KEY,
                stage      VARCHAR NOT NULL,
                batch_id   VARCHAR,
                run_id     VARCHAR,
                row_json   JSON,
                error_type VARCHAR NOT NULL,
                error_msg  VARCHAR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        run(con, "CREATE SEQUENCE IF NOT EXISTS ops.dlq_seq START 1")
        run(con, """
            CREATE TABLE IF NOT EXISTS ops.audit_runs (
                run_id        VARCHAR NOT NULL,
                batch_id      VARCHAR,
                stage         VARCHAR NOT NULL,
                status        VARCHAR NOT NULL,
                rows_in       BIGINT,
                rows_upserted BIGINT,
                rows_deduped  BIGINT,
                rows_late     BIGINT,
                rows_dlq      BIGINT,
                duration_ms   BIGINT,
                metrics       JSON,
                ts            TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
        run(con, """
            CREATE TABLE IF NOT EXISTS ops.snapshot_meta (
                run_id         VARCHAR NOT NULL,
                snapshot_table VARCHAR NOT NULL,
                target_table   VARCHAR NOT NULL,
                rows           BIGINT,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            )""")
    finally:
        con.close()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:8]


def audit(con: duckdb.DuckDBPyConnection, run_id: str, batch_id: str | None, stage: str,
          status: str, *, rows_in: int | None = None, rows_upserted: int | None = None,
          rows_deduped: int | None = None, rows_late: int | None = None,
          rows_dlq: int | None = None, duration_ms: int | None = None,
          metrics: dict[str, Any] | None = None) -> None:
    run(con, 
        "INSERT INTO ops.audit_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())",
        [run_id, batch_id, stage, status, rows_in, rows_upserted, rows_deduped,
         rows_late, rows_dlq, duration_ms, json.dumps(metrics or {}, ensure_ascii=False)])


def dlq_add(con: duckdb.DuckDBPyConnection, stage: str, batch_id: str | None,
            run_id: str, rows: list[dict[str, Any]], error_type: str,
            error_msg: str) -> int:
    if not rows:
        return 0
    seq = run(con, "SELECT nextval('ops.dlq_seq') FROM range(?)", [len(rows)])
    for dlq_id, row in zip([r[0] for r in seq], rows):
        run(con, 
            "INSERT INTO ops.dlq VALUES (?, ?, ?, ?, ?, ?, ?, now())",
            [dlq_id, stage, batch_id, run_id, json.dumps(row, ensure_ascii=False, default=str),
             error_type, error_msg])
    return len(rows)


def dlq_count(con: duckdb.DuckDBPyConnection) -> int:
    return int(run(con, "SELECT count(*) FROM ops.dlq")[0][0])


def upsert_raw(con: duckdb.DuckDBPyConnection, batch: pd.DataFrame, batch_id: str,
               run_id: str, watermark: pd.Timestamp | None = None) -> dict[str, Any]:
    """Идемпотентная загрузка батча в raw.events через MERGE по idempotency_key.

    Возвращает метрики: rows_in, inserted (новые), updated (повторы — доказательство
    идемпотентности при re-run), deduped (дубли внутри батча), late (поздние по watermark).
    """
    rows_in = len(batch)
    df = batch.copy()
    if "idempotency_key" not in df.columns:
        df["idempotency_key"] = make_idempotency_key(df)
    before = len(df)
    df = df.drop_duplicates(subset="idempotency_key", keep="first")
    deduped = before - len(df)

    if watermark is not None:
        df["is_late"] = df["event_time"] < pd.Timestamp(watermark)
    else:
        df["is_late"] = False
    df["batch_id"] = batch_id

    con.register("batch_view", df[["idempotency_key", "batch_id", "event_time", "is_late"]
                                   + FEATURE_COLUMNS + KEY_COLUMNS])
    cols = [quote(c) for c in FEATURE_COLUMNS + KEY_COLUMNS]
    col_list = ", ".join(cols)
    # Предподсчёт совпадений (MERGE в duckdb без RETURNING merge_action):
    # matched → UPDATE (доказательство повторной доставки), остальные → INSERT.
    matched = int(run(con, """
        SELECT count(*) FROM batch_view s
        WHERE EXISTS (SELECT 1 FROM raw.events t WHERE t.idempotency_key = s.idempotency_key)
        """)[0][0])
    run(con, f"""
        MERGE INTO raw.events AS t
        USING (SELECT * FROM batch_view) AS s
        ON t.idempotency_key = s.idempotency_key
        WHEN MATCHED THEN UPDATE SET ingested_at = now(), is_late = s.is_late
        WHEN NOT MATCHED THEN INSERT (idempotency_key, batch_id, event_time, is_late,
                                      {col_list})
             VALUES (s.idempotency_key, s.batch_id, s.event_time, s.is_late,
                     {', '.join('s.' + c for c in cols)})""")
    con.unregister("batch_view")
    inserted = len(df) - matched
    updated = matched
    late = int(df["is_late"].sum())
    return {"rows_in": rows_in, "inserted": inserted, "updated": updated,
            "deduped_intra_batch": deduped, "late": late}


def snapshot_raw(con: duckdb.DuckDBPyConnection, run_id: str) -> str:
    """Полная копия raw.events как снапшот версии (аналог Delta Time Travel,
    реализованный на доступном стеке; стоимость полного копирования — в отчёте)."""
    snap = f"snapshots.raw_events_{run_id.replace('-', '_')}"
    run(con, f"CREATE OR REPLACE TABLE {snap} AS SELECT * FROM raw.events")
    rows = int(run(con, f"SELECT count(*) FROM {snap}")[0][0])
    run(con, "INSERT INTO ops.snapshot_meta VALUES (?, ?, 'raw.events', ?, now())",
                [run_id, snap, rows])
    return snap


def rollback_raw(con: duckdb.DuckDBPyConnection, snapshot_table: str) -> int:
    """Откат raw.events к снапшоту: атомарная замена таблицы."""
    run(con, f"CREATE OR REPLACE TABLE raw.events AS SELECT * FROM {snapshot_table}")
    return int(run(con, "SELECT count(*) FROM raw.events")[0][0])


def watermark(con: duckdb.DuckDBPyConnection, allowed_lateness_s: float) -> pd.Timestamp | None:
    """watermark = max(event_time) - allowed_lateness; события раньше — late-arriving."""
    rows = run(con, f"""
        SELECT CASE WHEN max(event_time) IS NULL THEN NULL
                    ELSE max(event_time) - INTERVAL {int(allowed_lateness_s)} SECOND
               END FROM raw.events""")
    row = rows[0]
    return None if row[0] is None else pd.Timestamp(row[0])
