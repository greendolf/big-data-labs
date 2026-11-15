"""ЛР3 (часть 7): мониторинг и отчётность — графики из таблиц аудита."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lab3 import warehouse as wh  # noqa: E402

RESULTS = Path("lab3/results")
SLA_MS = 50  # из задания; проверяем честно на micro-batch эмуляции


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def build_charts(warehouse: Path, out_dir: Path | None = None) -> dict[str, str]:
    out = out_dir or RESULTS
    out.mkdir(parents=True, exist_ok=True)
    con = wh.connect(warehouse)
    paths: dict[str, str] = {}
    try:
        # 1. Задержка обработки батча против SLA из задания
        # задержка события = длительность его батча (micro-batch семантика)
        rows = wh.run(con, """
            SELECT batch_id, duration_ms FROM ops.audit_runs
            WHERE stage = 'load_raw' AND status = 'ok' AND batch_id IS NOT NULL
              AND run_id NOT LIKE 'canary%'
            ORDER BY ts""")
        batches = [r[0] for r in rows]
        batch_ms = [float(r[1]) for r in rows]
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.bar(batches, batch_ms, width=0.7, color="#4878b0")
        ax.axhline(SLA_MS, color="#d1495b", linestyle="--", lw=1.2,
                   label=f"SLA из задания: {SLA_MS} мс на событие (= время его батча)")
        for s in ax.spines.values():
            s.set_visible(False)
        _style(ax, "Задержка события при micro-batch: длительность обработки батча vs SLA", "мс/батч")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        p = str(out / "monitoring_latency_vs_sla.png")
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["latency"] = p

        # 2. DLQ и late-arriving по батчам
        dlq = wh.run(con, """
            SELECT batch_id, sum(rows_dlq), sum(rows_late) FROM ops.audit_runs
            WHERE batch_id IS NOT NULL GROUP BY batch_id ORDER BY batch_id""")
        ids = [r[0] for r in dlq]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
        a1.bar(ids, [r[1] for r in dlq], color="#d1495b")
        _style(a1, "Строки в DLQ по батчам (включая фикстуры ошибок)", "строк")
        a2.bar(ids, [r[2] for r in dlq], color="#54b683")
        _style(a2, "Late-arriving события по батчам (watermark − 4 ч)", "строк")
        a2.tick_params(axis="x", labelsize=6)
        for s in a1.spines.values():
            s.set_visible(False)
        for s in a2.spines.values():
            s.set_visible(False)
        p = str(out / "monitoring_dlq_late.png")
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["dlq_late"] = p

        # 3. Распределение риск-скора витрины
        risks = wh.run(con, """
            SELECT risk_score FROM dbt_marts.dim_device_profile""")
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.hist([r[0] for r in risks], bins=25, color="#6f5fa3")
        _style(ax, "Распределение risk_score устройств (витрина dim_device_profile)",
               "устройств")
        for s in ax.spines.values():
            s.set_visible(False)
        p = str(out / "monitoring_risk_hist.png")
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["risk_hist"] = p
    finally:
        con.close()
    return paths


def sla_summary(warehouse: Path) -> dict[str, float | bool | None]:
    """Задержка события при micro-batch = время обработки его батча (событие ждёт,
    пока обработают весь батч). Считаем p50/p95 от batch-duration и вердикт по SLA 50 мс;
    амортифицированная стоимость на событие приводится отдельно — это НЕ задержка."""
    import numpy as np

    con = wh.connect(warehouse)
    try:
        rows = wh.run(con, """
            SELECT l.duration_ms, e.rows_in
            FROM ops.audit_runs l
            JOIN ops.audit_runs e
              ON e.batch_id = l.batch_id AND e.stage = 'extract'
            WHERE l.stage = 'load_raw' AND l.status = 'ok' AND l.batch_id IS NOT NULL
              AND l.run_id NOT LIKE 'canary%'""")
    finally:
        con.close()
    if not rows:
        return {"p50_ms": None, "p95_ms": None, "sla_50ms_passed": None}
    dur = np.array([r[0] for r in rows], dtype=float)
    p50, p95 = float(np.percentile(dur, 50)), float(np.percentile(dur, 95))
    amortized = float(dur.sum() / max(sum(r[1] for r in rows), 1))
    return {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
            "amortized_ms_per_event": round(amortized, 4),
            "sla_50ms_passed": bool(p95 <= SLA_MS), "n_batches": len(rows)}
