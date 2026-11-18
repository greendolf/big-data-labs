"""ЛР4: построение feature table для Feast (агрегаты активности устройств).

Офлайн/онлайн-согласованность строится на маленьком детерминированном окне:
для каждой пары (Device_Name, час) считаем почасовые агрегаты. Таблица небольшая
(9 устройств × 24 часа), но это настоящий Feature View с ttl — тот же источник
кормит и get_historical_features (обучение), и get_online_features (инференс).

Важно: метка target и «будущие» признаки в таблицу НЕ попадают — только то, что
знаваемо на момент конца часа (иначе leakage при офлайн-джойне).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "data" / "lab4" / "feature_store"
REPO = ROOT / "lab4" / "feast_repo"

# признаки, доступные «как есть» на момент часа. Состава метки здесь НЕТ:
# attack_rate/subtype-подобные агрегаты напрямую кодируют label (в том числе
# текущего события) — при point-in-time джойне это утечка, AUC искусственно 1.0.
AGG_METRICS = [
    ("event_count_1h", "COUNT(*)::DOUBLE"),
    ("avg_magnitude_1h", 'AVG("HH_L0.1_magnitude")'),
    ("avg_radius_1h", 'AVG("HH_L0.1_radius")'),
    ("avg_mi_mean_1h", 'AVG("MI_dir_L0.1_mean")'),
    ("std_mi_mean_1h", 'STDDEV("MI_dir_L0.1_mean")'),
    ("avg_weight_1h", 'AVG("HH_L0.1_weight")'),
    ("p95_mag_1h", 'QUANTILE_CONT("HH_L0.1_magnitude", 0.95)'),
]


def build_feature_table() -> pd.DataFrame:
    """raw.events → hourly device aggregates (parquet для Feast FileSource)."""
    OUT.mkdir(parents=True, exist_ok=True)
    lab3_wh = ROOT / "data" / "lab3" / "warehouse.duckdb"
    con = duckdb.connect(str(lab3_wh), read_only=True)
    cols = ",\n".join(f'{expr} AS "{name}"' for name, expr in AGG_METRICS)
    df = con.execute(f"""
        SELECT Device_Name,
               date_trunc('hour', event_time) AS event_timestamp,
               {cols}
        FROM raw.events
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    con.close()
    df["created"] = pd.Timestamp.utcnow().tz_localize(None)
    # Feast требует tz-aware? duckdb отдаёт naive — нормализуем к UTC-naive (ок для file source)
    path = OUT / "device_hourly_stats.parquet"
    df.to_parquet(path, index=False)
    # дубликат в feast_repo/ — FileSource ищет относительно repo (для apply)
    df.to_parquet(REPO / "device_hourly_stats.parquet", index=False)
    return df


def entity_frame(df_events: pd.DataFrame, sample: int = 4000,
                 seed: int = 3) -> pd.DataFrame:
    """entity_df для point-in-time join: случайные события (device, timestamp)."""
    return (df_events[["Device_Name", "event_time"]]
            .rename(columns={"event_time": "event_timestamp"})
            .sample(min(sample, len(df_events)), random_state=seed)
            .reset_index(drop=True))


FEATURE_NAMES = [name for name, _ in AGG_METRICS]
