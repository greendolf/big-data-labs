"""ЛР3: оркестрация Prefect 3 (часть 4 задания).

Задачи оборачивают чистые функции pipeline/quality/dbt_runner — один и тот же код
идёт и в ручные прогоны, и в оркестратор (проверяемость). Извлекаемость свойств:
  * retries + экспоненциальная задержка на extract (список задержек [1, 2, 4] c);
  * явные зависимости extract → contract/validate → load → dbt → marts-гейт → alerts;
  * динамическое ветвление: alert-задача маппится по devices с risk_score >= 0.7;
  * метрики каждой стадии → ops.audit_runs (отчётность/мониторинг).

PREFECT_HOME и API URL задаются окружением (см. run_notebook.py / runner), чтобы
UI сервер видел прогоны.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
os.environ.setdefault("PREFECT_HOME", str(ROOT / "data" / "lab3" / "prefect"))

from prefect import flow, get_run_logger, task  # noqa: E402

from lab3 import warehouse as wh  # noqa: E402
from lab3.dbt_runner import run_dbt  # noqa: E402
from lab3.pipeline import process_batch  # noqa: E402


@task(name="process-batch", retries=2, retry_delay_seconds=[1, 2, 4],
      tags=["etl", "fraud"])
def task_process_batch(batch_file: str, batch_id: str, *, warehouse: str,
                       flaky_failures: int = 0, run_id: str = "") -> dict[str, Any]:
    log = get_run_logger()
    alerts: list[str] = []
    metrics = process_batch(Path(batch_file), batch_id, run_id=run_id or None,
                            warehouse=Path(warehouse),
                            flaky_failures=flaky_failures,
                            alert_hook=lambda m: (alerts.append(m), log.warning(m)))
    metrics["alerts"] = alerts
    return metrics


@task(name="dbt-build", retries=1, retry_delay_seconds=5, tags=["elt"])
def task_dbt_build(warehouse: str) -> dict[str, Any]:
    result = run_dbt("build", "--select", "stg_events fct_attack_events dim_device_profile",
                     warehouse=Path(warehouse))
    if result["returncode"] != 0:
        raise RuntimeError(f"dbt build failed rc={result['returncode']}: {result['stderr_tail'][-500:]}")
    payload = result.get("result") or {}
    return {"returncode": result["returncode"],
            "results": [{"name": r.get("unique_id"), "status": r.get("status")}
                        for r in payload.get("results", [])]}


@task(name="dbt-snapshot", tags=["time-travel"])
def task_dbt_snapshot(warehouse: str) -> dict[str, Any]:
    result = run_dbt("snapshot", warehouse=Path(warehouse))
    if result["returncode"] != 0:
        raise RuntimeError(f"dbt snapshot failed: {result['stderr_tail'][-500:]}")
    return {"returncode": 0}


@task(name="marts-gate", tags=["quality"])
def task_marts_gate(warehouse: str) -> dict[str, Any]:
    """GE-гейт по витрине профилей (после трансформаций)."""
    from lab3.quality import get_context, validate_dataframe
    con = wh.connect(Path(warehouse))
    try:
        df = con.execute("SELECT * FROM dbt_marts.dim_device_profile").df()
    finally:
        con.close()
    summary = validate_dataframe(get_context(), df, kind="marts", score_threshold=0.999)
    if not summary["passed"]:
        raise RuntimeError(f"marts quality gate failed: {summary['failures'][:3]}")
    return summary


@task(name="risk-alerts", tags=["monitoring"])
def task_risk_alert(warehouse: str, device: str, risk: float, top_subtype: str) -> str:
    log = get_run_logger()
    msg = f"ALERT: устройство {device} risk_score={risk:.2f} (доминанта {top_subtype})"
    log.warning(msg)
    return msg


@flow(name="fraud-batch", log_prints=True)
def flow_fraud_batch(batch_file: str, batch_id: str, *, warehouse: str,
                     flaky_failures: int = 0, run_id: str = "") -> dict[str, Any]:
    metrics = task_process_batch(batch_file, batch_id, warehouse=warehouse,
                                 flaky_failures=flaky_failures, run_id=run_id)
    return metrics


@flow(name="fraud-nightly", log_prints=True)
def flow_fraud_nightly(manifest_batches: list[dict[str, Any]], *, warehouse: str,
                       flaky_at: int = -1) -> dict[str, Any]:
    """Ночной прогон: батчи последовательно (один писатель duckdb), затем ELT-слой
    и витринный гейт; алерты — динамический map по «горячим» устройствам."""
    from lab3.pipeline import reset_attempts
    reset_attempts()
    totals = {"batches": 0, "inserted": 0, "updated": 0, "late": 0, "dlq": 0, "deduped": 0}
    for i, b in enumerate(manifest_batches):
        m = flow_fraud_batch(b["file"], b["batch_id"], warehouse=warehouse,
                             flaky_failures=(1 if i == flaky_at else 0), run_id="")
        totals["batches"] += 1
        totals["inserted"] += m.get("inserted", 0)
        totals["updated"] += m.get("updated", 0)
        totals["late"] += m.get("late", 0)
        totals["dlq"] += m.get("dlq", 0)
        totals["deduped"] += m.get("deduped_intra_batch", 0)
    dbt_res = task_dbt_build(warehouse)
    snap_res = task_dbt_snapshot(warehouse)
    gate = task_marts_gate(warehouse)

    con = wh.connect(Path(warehouse))
    try:
        hot = con.execute("""
            select device_name, risk_score, top_subtype from dbt_marts.dim_device_profile
            where risk_score >= 0.7 order by risk_score desc limit 10""").fetchall()
    finally:
        con.close()
    alerts = task_risk_alert.map(warehouse, [d for d, _, _ in hot],
                                 [r for _, r, _ in hot], [s for _, _, s in hot])
    totals["dbt"] = dbt_res
    totals["snapshot"] = snap_res
    totals["marts_gate"] = gate
    totals["alerts"] = [f.result() for f in alerts]
    return totals
