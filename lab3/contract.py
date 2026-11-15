"""ЛР3: контракт схемы + построчная валидация с маршрутизацией ошибок в DLQ.

Два уровня защиты данных:
  1) Контракт схемы (батч-уровень): переименования лечатся алиас-мапой (fallback на
     предыдущую схему), неизвестные колонки отбрасываются с алертом, отсутствие
     обязательной колонки — SchemaDriftError (батч не загружается).
  2) Построчная валидация: правила-функции; строки-нарушители уходят в ops.dlq,
     чистые — в загрузку. Так один битый поток не роняет пайплайн (criterion «DLQ»).
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from lab3.warehouse import (FEATURE_COLUMNS, KEY_COLUMNS, LABEL_SET, NONNEGATIVE_COLUMNS,
                            RAW_COLUMNS, quote)  # noqa: F401 (quote re-export для тестов)


class SchemaDriftError(RuntimeError):
    pass


# Fallback-мапа: при переименовании колонки источник «старой» схемы восстанавливается
# из нового имени — пайплайн продолжает работать на предыдущей схеме.
ALIAS_MAP: dict[str, str] = {"HH_L0.1_SD": "HH_L0.1_std", "device_name": "Device_Name"}
REQUIRED_COLUMNS = ["event_time", *KEY_COLUMNS]


def apply_schema_contract(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Возвращает (df в контрактной схеме, отчёт о дрейфе). Бросает SchemaDriftError,
    если дрейф не восстанавливается алиасами (не хватает обязательной колонки)."""
    report: dict[str, Any] = {"renamed_back": {}, "dropped_unknown": [], "missing": []}
    out = df.copy()

    for new_name, canonical in ALIAS_MAP.items():
        if new_name in out.columns and canonical not in out.columns:
            out = out.rename(columns={new_name: canonical})
            report["renamed_back"][new_name] = canonical

    missing = [c for c in REQUIRED_COLUMNS + FEATURE_COLUMNS if c not in out.columns]
    unknown = [c for c in out.columns if c not in REQUIRED_COLUMNS + RAW_COLUMNS
               and c not in ("idempotency_key", "batch_id", "is_late")]
    report["missing"] = missing
    report["dropped_unknown"] = unknown
    if missing:
        raise SchemaDriftError(f"schema drift: отсутствуют обязательные колонки {missing}")
    if unknown:
        out = out.drop(columns=unknown)
    return out, report


# --- построчные правила ---------------------------------------------------------

def _not_null(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].notna() & (df[col].astype("string").str.len() > 0)


def row_rules() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    """(rule_id, error_type, маска_валидности). Строка попадает в DLQ по первому нарушенному правилу."""
    rules: list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("key_not_null", "null_key",
         lambda d: _not_null(d, "Device_Name") & _not_null(d, "Attack") & _not_null(d, "Attack_subType")),
        ("label_domain", "label_out_of_domain", lambda d: d["label"].isin(LABEL_SET)),
        ("variance_nonneg", "negative_variance",
         lambda d: d[[c for c in NONNEGATIVE_COLUMNS]].notna().all(axis=1)
         & (d[[c for c in NONNEGATIVE_COLUMNS]] >= 0).all(axis=1)),
    ]
    return rules


def split_clean_and_bad(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(чистые, грязные+s колонки _rule/_error_type, список сработавших правил)."""
    valid = pd.Series(True, index=df.index)
    bad_parts: list[pd.DataFrame] = []
    tripped: list[str] = []
    for rule_id, error_type, mask_fn in row_rules():
        mask = mask_fn(df)
        invalid = ~mask & valid  # только по ещё не помеченным — одно правило на строку
        if invalid.any():
            part = df[invalid].copy()
            part["_rule"] = rule_id
            part["_error_type"] = error_type
            bad_parts.append(part)
            tripped.append(f"{rule_id}:{int(invalid.sum())}")
            valid &= mask
    bad = pd.concat(bad_parts, ignore_index=True) if bad_parts else df.iloc[0:0].copy()
    return df[valid].copy(), bad, tripped


def bad_rows_grouped_by_error(bad: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """{error_type: [row dicts + _rule]} — для пакетной записи в ops.dlq по типам ошибок."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in bad.to_dict(orient="records"):
        etype = str(row.pop("_error_type", "unknown"))
        grouped.setdefault(etype, []).append(row)  # _rule остаётся внутри row_json
    return grouped
