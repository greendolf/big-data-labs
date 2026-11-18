"""ЛР4: мониторинг данных и моделей (PSI, KS, Evidently, алерты, retrain-логика).

Пороги (канон для PSI): <0.1 нет дрейфа, 0.1–0.25 умеренный, >0.25 сильный.
DATA_DRIFT_SCORE = доля колонок с PSI > 0.2 (канон Evidently — доля дрейфовавших).
Отчёт Evidently — визуальный артефакт; ЧИСЛА метрик считаем сами (детерминизм
и тестопригодность), и это честно зафиксировано в отчёте.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path.cwd()
ALERT_LOG = ROOT / "data" / "lab4" / "alerts.log"

PSI_DRIFT = 0.2          # порог «колонка дрейфует»
DRIFT_ALERT = 0.2        # порог алерта по доле дрейфовавших колонок
RETRAIN_DRIFT = 0.3      # порог триггера переобучения по дрейфу
AUC_DROP_ALERT = 0.1     # алерт по падению AUC
RETRAIN_AUC_DROP = 0.15  # триггер переобучения по падению качества


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index по квантильным корзинам эталона."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    e_cnt, _ = np.histogram(expected, bins=qs)
    a_cnt, _ = np.histogram(actual, bins=qs)
    e = np.clip(e_cnt / max(e_cnt.sum(), 1), 1e-6, None)
    a = np.clip(a_cnt / max(a_cnt.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ks_2samp

    r = ks_2samp(expected, actual)
    return float(r.statistic), float(r.pvalue)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        e = reference[c].to_numpy(dtype="float64")
        a = current[c].to_numpy(dtype="float64")
        p = psi(e, a)
        ks_stat, ks_p = ks(e, a)
        rows.append({"column": c, "psi": p, "ks": ks_stat, "ks_pvalue": ks_p,
                     "drifted": p > PSI_DRIFT})
    return pd.DataFrame(rows)


def drift_score(report: pd.DataFrame) -> float:
    """Доля дрейфовавших колонок — то, что отдаём как DATA_DRIFT_SCORE."""
    return float(report["drifted"].mean())


def evidently_artifact(reference: pd.DataFrame, current: pd.DataFrame,
                       out_dir: Path) -> dict[str, Any]:
    """HTML+JSON отчёт Evidently (DataDriftPreset) — как требует задание."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snap = Report([DataDriftPreset()]).run(current, reference)
    snap.save_html(str(out_dir / "evidently_drift.html"))
    snap.save_json(str(out_dir / "evidently_drift.json"))
    return {"html": str(out_dir / "evidently_drift.html"),
            "json": str(out_dir / "evidently_drift.json")}


def send_alert(message: str) -> None:
    """Алерт: локальный журнал (всегда) + Slack webhook, если задан SLACK_WEBHOOK_URL.

    Честно: без настроенного вебхука интеграция остаётся заглушкой-интерфейсом —
    это НЕ выдаём за « боевой Slack».
    """
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}\n"
    with ALERT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    url = __import__("os").environ.get("SLACK_WEBHOOK_URL")
    if url:
        import requests

        requests.post(url, json={"text": f"🚨 ML MONITORING ALERT: {message}"},
                      timeout=5)


def check_retrain_needed(drift_sc: float, auc_drop: float,
                         days_since_retrain: float,
                         weekly_days: float = 7.0) -> tuple[bool, str]:
    """Детерминированная три-проверка из задания: дрейф / падение / календарь."""
    if drift_sc > RETRAIN_DRIFT:
        return True, f"drift_score {drift_sc:.3f} > {RETRAIN_DRIFT}"
    if auc_drop > RETRAIN_AUC_DROP:
        return True, f"падение AUC {auc_drop:.3f} > {RETRAIN_AUC_DROP}"
    if days_since_retrain >= weekly_days:
        return True, f"еженедельный retrain ({days_since_retrain:.1f} дн.)"
    return False, "пороги не превышены"
