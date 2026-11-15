"""ЛР3 (часть 7): модель канарейки и отката для ELT-слоя.

Канарейка = тот же dbt-проект, но target=canary: изолированная схема dbt_canary_* и
изменённая бизнес-логика через dbt-var (вес тяжести риск-скоринга). Прогон гоняется
по ТЕМ ЖЕ сырым данным, затем smoke-гейт (GE по витрине канарейки). Промоушен —
атомарная замена витрины + пред-снапшот; откат — возврат пред-снапшота.
Всё честным образом локальная модель: GitHub PR-машина в вуде не крутится, но шаги
промоушена/отката — реальные SQL-операции над warehouse.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab3 import warehouse as wh
from lab3.dbt_runner import run_dbt

STABLE_DIM = "dbt_marts.dim_device_profile"
CANARY_DIM = "dbt_canary_marts.dim_device_profile"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def run_canary(warehouse: Path, severity_weight: float = 0.6) -> dict[str, Any]:
    res = run_dbt("build", "--exclude", "device_profile_history",
                  warehouse=warehouse, target="canary",
                  vars={"risk_severity_weight": severity_weight})
    con = wh.connect(warehouse)
    try:
        stats = wh.run(con, f"""
            SELECT count(*), round(avg(risk_score), 4), max(risk_score)
            FROM {CANARY_DIM}""")[0]
    finally:
        con.close()
    return {"dbt_rc": res["returncode"], "devices": int(stats[0]),
            "avg_risk": float(stats[1]) if stats[1] is not None else None,
            "max_risk": float(stats[2]) if stats[2] is not None else None,
            "severity_weight": severity_weight}


def smoke_gate(warehouse: Path) -> dict[str, Any]:
    """Smoke-тест канареечной витрины: GE-гейт + базовые инварианты."""
    import pandas as pd

    from lab3.quality import get_context, validate_dataframe
    con = wh.connect(warehouse)
    try:
        df = con.execute(f"SELECT * FROM {CANARY_DIM}").df()
    finally:
        con.close()
    summary = validate_dataframe(get_context(), df, kind="marts", score_threshold=0.999)
    summary["non_empty"] = len(df) > 0
    summary["passed"] = summary["passed"] and summary["non_empty"]
    return summary


def promote_canary(warehouse: Path) -> dict[str, Any]:
    """Промоушен: снимок стабильной витрины → атомарная замена на канарейку."""
    con = wh.connect(warehouse)
    try:
        backup = f"snapshots.dim_device_profile_prev_{_ts()}"
        wh.run(con, f"CREATE SCHEMA IF NOT EXISTS snapshots")
        wh.run(con, f"CREATE TABLE {backup} AS SELECT * FROM {STABLE_DIM}")
        wh.run(con, f"CREATE OR REPLACE TABLE {STABLE_DIM} AS SELECT * FROM {CANARY_DIM}")
        wh.audit(con, "canary", None, "promote", "ok", metrics={"backup": backup})
        return {"backup_table": backup, "promoted": True}
    finally:
        con.close()


def rollback_canary(warehouse: Path, backup_table: str) -> dict[str, Any]:
    con = wh.connect(warehouse)
    try:
        wh.run(con, f"CREATE OR REPLACE TABLE {STABLE_DIM} AS SELECT * FROM {backup_table}")
        rows = int(wh.run(con, f"SELECT count(*) FROM {STABLE_DIM}")[0][0])
        wh.audit(con, "canary", None, "rollback", "ok", metrics={"restored_from": backup_table})
        return {"restored_from": backup_table, "rows": rows}
    finally:
        con.close()
