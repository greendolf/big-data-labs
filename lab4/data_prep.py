"""ЛР4: подготовка ML-датасета из хранилища ЛР3 (Fraud Detection).

Стратификация по времени (replay-модель event_time из ЛР3, окно — 24 часа),
чтобы сплиты и сравнение batch/online были честными во времени:
  train  = первые 18 часов,
  test   = последние 6 часов (без утечки будущего),
  drift-окно — отдельная фикстура (часть мониторинга).

Строки с known-bad (DLQ-кандидаты) в хранилище уже отсутствуют; здесь только
label по признаку атаки (Attack != 'Normal'), метки совпадают с витринами ЛР3.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path.cwd()
LAB3_WH = ROOT / "data" / "lab3" / "warehouse.duckdb"
OUT = ROOT / "data" / "lab4"

FEATURES = [
    "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std", "HH_L0.1_magnitude",
    "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
    "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std", "HpHp_L0.1_magnitude",
    "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc",
]


def build_datasets() -> dict[str, int]:
    """raw.events → events.parquet (float32) + сплит-манифест. Идемпотентно."""
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(LAB3_WH), read_only=True)
    # float32 для ML (−0.4% точности, −50% памяти); event_time для честного сплита
    cols = ", ".join(f'"{c}"::FLOAT AS "{c}"' for c in FEATURES)
    con.execute(f"""
        COPY (
            SELECT idempotency_key, Device_Name, Attack, label::INT AS label,
                   event_time, batch_id, is_late, {cols}
            FROM raw.events
        ) TO '{(OUT / 'events.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    manifest = con.execute("""
        SELECT count(*) n, min(event_time), max(event_time), count(DISTINCT Device_Name)
        FROM raw.events
    """).fetchone()
    n, t_min, t_max, devices = manifest
    n_train = con.execute("""
        SELECT count(*) FROM raw.events
        WHERE event_time < TIMESTAMP '2026-11-01 18:00:00'
    """).fetchone()[0]
    con.close()
    stats = {"rows": int(n), "devices": int(devices), "train_rows": int(n_train),
             "test_rows": int(n) - int(n_train)}
    pd.DataFrame([{"k": k, "v": v} for k, v in stats.items()]).to_csv(
        OUT / "dataset_stats.csv", index=False)
    return stats


def load_events() -> pd.DataFrame:
    return pd.read_parquet(OUT / "events.parquet")


def time_split(df: pd.DataFrame, cutoff: str = "2026-11-01 18:00:00"):
    """Честный temporal split: train строго раньше test по event_time.

    Replay-окно ЛР3 — 24 часа; train = первые 18 ч (~75%), test = последние 6 ч.
    """
    is_train = df["event_time"] < pd.Timestamp(cutoff)
    return df[is_train], df[~is_train]


_DRIFT_SCALED = [  # «новая волна»: профиль устройств интенсивнее по всей семье
    "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_mean", "HH_L0.1_std",
    "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_mean", "HpHp_L0.1_std",
]


def make_drift_window(df_test: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Фикстура concept drift для мониторинга: «новая волна атак» меняет и
    распределение, и связь признаков с меткой:
      1) сдвиг масштаба семьи intensity-признаков (+45% — новый профиль устройств),
      2) star-sign flip HH_L0.1_magnitude у 15% строк (нормальные события уезжают
         в «атакующий» хвост — старая модель начинает ошибаться).
    Окно отдельное — модель его НЕ видела при обучении. Фикстура детерминирована.
    """
    import numpy as np

    g = np.random.default_rng(seed)
    drift = df_test.reset_index(drop=True).copy()   # порядок по времени сохраняем
    for c in _DRIFT_SCALED:
        drift[c] = drift[c] * 1.45
    flip = g.random(len(drift)) < 0.15
    mag = drift.loc[flip, "HH_L0.1_magnitude"]
    drift.loc[flip, "HH_L0.1_magnitude"] = mag.max() - mag  # инверсия связи
    drift["is_drifted"] = True
    return drift


if __name__ == "__main__":
    print(build_datasets())
