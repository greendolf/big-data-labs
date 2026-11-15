"""ЛР3: нарезка cleaned-датасета на микро-батчи + детерминированные фикстуры сбоев.

Модель времени (ЧЕСТНО): в исходном IDS-датасете нет event timestamp (зафиксировано
в ЛР1), поэтому event_time — ЯВНО ЗАДАННАЯ replay-модель: монотонные «часы» от t0 с
детерминированным джиттером по хэшу строки. Это модель нагрузки для проверки свойств
пайплайна (идемпотентность, late-arriving, SLA-замеры), а не утверждение о реальных
времени/частотах датасета.

Всё детерминированно от seed: повторный запуск даёт побитово те же батчи.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from lab3.warehouse import FEATURE_COLUMNS, KEY_COLUMNS, quote

T0 = "2026-11-01 00:00:00"
WINDOW_SECONDS = 24 * 3600  # сутки «нагрузки» на весь набор батчей


def _all_cols() -> list[str]:
    return FEATURE_COLUMNS + KEY_COLUMNS


def build_batches(source_parquet: Path, out_dir: Path, n_batches: int = 20,
                  seed: int = 774, late_share: float = 0.01) -> list[dict[str, Any]]:
    """Чистый набор → n_batches parquet-файлов + манифест.

    - порядок строк: md5 от кортежа (детерминированный «перемешанный»);
    - event_time: t0 + row_no*шаг + джиттер (hash(md5)%1000/1000 * шаг);
    - late-arriving: ~late_share строк батча i переносятся в батч i+2 БЕЗ смены
      event_time — они придут после watermark и должны корректно догнать витрины.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src = str(Path(source_parquet).resolve()).replace("'", "''")
    cols = ", ".join(quote(c) for c in _all_cols())
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            WITH base AS (
                SELECT {cols},
                       md5(concat_ws(chr(31), {cols})) AS h
                FROM read_parquet('{src}')
            ), ordered AS (
                SELECT *, row_number() OVER (ORDER BY h) AS row_no,
                       count(*) OVER () AS total
                FROM base
            )
            SELECT {cols}, h, row_no, total,
                   TIMESTAMP '{T0}'
                     + ((row_no * {WINDOW_SECONDS} / total)::BIGINT) * INTERVAL 1 SECOND
                     + (((hash(h) % 1000 + 1000) % 1000) * INTERVAL 1 MILLISECOND) AS event_time
            FROM ordered ORDER BY row_no""").df()
    finally:
        con.close()

    total = len(df)
    edges = [round(i * total / n_batches) for i in range(n_batches + 1)]
    chunks = [df.iloc[edges[i]:edges[i + 1]].copy() for i in range(n_batches)]

    # Перенос поздних строк: i → i+2 (без смены event_time → они станут late-arriving).
    injected = 0
    for i in range(n_batches - 2):
        chunk = chunks[i]
        mask = chunk["h"].map(lambda h: int(h[:4], 16) % 100 < round(late_share * 100))
        movers = chunk[mask]
        if len(movers):
            chunks[i] = chunk[~mask]
            chunks[i + 2] = pd.concat([chunks[i + 2], movers], ignore_index=True)
            injected += len(movers)

    manifest: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        batch_id = f"b{i:03d}"
        path = out_dir / f"{batch_id}.parquet"
        chunk.drop(columns=["h", "total"]).to_parquet(path, index=False)
        manifest.append({
            "batch_id": batch_id, "file": str(path), "rows": len(chunk),
            "window_start": str(chunk["event_time"].min()),
            "window_end": str(chunk["event_time"].max()),
        })
    (out_dir / "manifest.json").write_text(
        json.dumps({"seed": seed, "n_batches": n_batches, "total_rows": total,
                    "late_rows_moved": injected, "batches": manifest},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def make_dirty(src_batch: Path, dst_batch: Path, n_null_key: int = 12,
               n_bad_label: int = 6, n_negative: int = 6) -> dict[str, int]:
    """Фикстура «грязных» строк: пустые ключи, label вне домена, отрицательная
    дисперсия. Должны быть перехвачены построчной валидацией и уйти в DLQ."""
    df = pd.read_parquet(src_batch)
    n = len(df)
    idx_null = df.index[:n_null_key]
    idx_label = df.index[n_null_key:n_null_key + n_bad_label]
    idx_neg = df.index[n_null_key + n_bad_label:n_null_key + 2 * n_bad_label]
    df.loc[idx_null, "Device_Name"] = None
    df.loc[idx_label, "label"] = 7
    df.loc[idx_neg, "MI_dir_L0.1_variance"] = -5.0
    df.to_parquet(dst_batch, index=False)
    assert len(df) == n
    return {"null_key": len(idx_null), "bad_label": len(idx_label),
            "negative_variance": len(idx_neg), "total_injected": n_null_key + 2 * n_bad_label}


def make_drift(src_batch: Path, dst_batch: Path) -> dict[str, Any]:
    """Фикстура schema drift: переименованная колонка (HH_L0.1_std → HH_L0.1_SD)
    и неизвестная новая колонка. Контракт обязан отловить; алиас-мапа даёт fallback."""
    df = pd.read_parquet(src_batch)
    df = df.rename(columns={"HH_L0.1_std": "HH_L0.1_SD"})
    df["new_signal_from_upstream"] = 42.0
    df.to_parquet(dst_batch, index=False)
    return {"renamed": {"HH_L0.1_std": "HH_L0.1_SD"}, "added": ["new_signal_from_upstream"]}
