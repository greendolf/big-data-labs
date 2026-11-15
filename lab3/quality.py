"""ЛР3: тестирование данных Great Expectations (часть 5 задания).

Персистентный file-context в lab3/great_expectations/: сьюты, definitions и checkpoint
хранятся на диске (идемпотентно: повторный запуск переиспользует), результаты прогонов
складываются в store, Data Docs (HTML) собираются в lab3/results/ge_docs.

GE вызывается как отдельная задача оркестратора дважды: ДО загрузки (батч) и ПОСЛЕ
трансформаций dbt (витрины). Порог quality_score — контракт гейта.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

import great_expectations as gx
from great_expectations.checkpoint import Checkpoint, UpdateDataDocsAction
from great_expectations.core.validation_definition import ValidationDefinition

from lab3.warehouse import ATTACK_DOMAIN, LABEL_SET, NONNEGATIVE_COLUMNS, quote  # noqa: F401

GE_ROOT = Path("lab3/great_expectations")


def get_context(root: Path | None = None):
    root = Path(root or GE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return gx.get_context(mode="file", context_root_dir=str(root.resolve()))


def _ensure_datasource(context):
    name = "fraud_batches"
    ds = next((d for d in context.data_sources.all().values() if d.name == name), None)
    if ds is None:
        ds = context.data_sources.add_pandas(name)
    if "batches" in ds.get_asset_names():
        asset = ds.get_asset("batches")
    else:
        asset = ds.add_dataframe_asset(name="batches")
    existing = {b.name for b in asset.batch_definitions}
    if "whole_batch" in existing:
        return asset.get_batch_definition("whole_batch")
    return asset.add_batch_definition_whole_dataframe("whole_batch")


def _names(collection) -> set[str]:
    items = collection.all()
    if isinstance(items, dict):
        return set(items.keys())
    return {getattr(x, "name", None) for x in items}


def _suite_sig(suite) -> list[str]:
    return sorted(f"{type(e).__name__}:{getattr(e, 'kwargs', {}).get('column')}"
                  for e in suite.expectations)


def _ensure_suite(context, name: str, expectations: list):
    new = gx.ExpectationSuite(name=name, expectations=expectations)
    if name not in _names(context.suites):
        context.suites.add(new)
    elif _suite_sig(context.suites.get(name)) != _suite_sig(new):
        context.suites.update(new)  # контракт изменился — обновляем сохранённый сьют
    return context.suites.get(name)


def _ensure_checkpoint(context, name: str, batch_def, suite_name: str, with_docs: bool):
    if name in _names(context.checkpoints):
        return context.checkpoints.get(name)
    vd_name = f"{name}__vd"
    if vd_name not in _names(context.validation_definitions):
        context.validation_definitions.add(ValidationDefinition(
            name=vd_name, data=batch_def, suite=suite_name))
    actions = [UpdateDataDocsAction(name="datadocs")] if with_docs else []
    return context.checkpoints.add(Checkpoint(
        name=name,
        validation_definitions=[context.validation_definitions.get(vd_name)],
        actions=actions))


def batch_expectations() -> list:
    e: list[Any] = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column=c)
        for c in ("Device_Name", "Attack", "Attack_subType", "label", "event_time")
    ]
    e += [
        gx.expectations.ExpectColumnValuesToBeInSet(column="label", value_set=LABEL_SET),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Attack", value_set=ATTACK_DOMAIN),
    ]
    e += [
        gx.expectations.ExpectColumnMinToBeBetween(column=c, min_value=0, strict_min=False)
        for c in NONNEGATIVE_COLUMNS
    ]
    e.append(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
    return e


def marts_expectations() -> list:
    return [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="device_name"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="device_name"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="attack_rate", min_value=0, max_value=1, strict_min=False, strict_max=False),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="events", min_value=1, strict_min=False),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1),
    ]


def _iter_suite_results(result):
    rr_list = result.run_results
    rr_iter = rr_list.values() if isinstance(rr_list, dict) else rr_list
    for rr in rr_iter:
        single = getattr(rr, "validation_result", None)
        if single is not None:
            yield single
            continue
        vr_list = getattr(rr, "expectation_suite_validation_result_list", None)
        if vr_list:
            yield from vr_list
            continue
        if hasattr(rr, "results"):  # сам ExpectationSuiteValidationResult
            yield rr


def _er_info(er) -> dict[str, Any]:
    column = None
    for attr in ("kwarg", "kwargs"):
        kw = getattr(er, attr, None)
        if isinstance(kw, dict) and "column" in kw:
            column = kw["column"]
            break
    if column is None:
        cfg = getattr(er, "expectation_config", None)
        if cfg is not None:
            column = getattr(cfg, "kwargs", {}).get("column")
    etype = getattr(er, "expectation_type", None)
    if not etype:
        cfg = getattr(er, "expectation_config", None)
        etype = getattr(cfg, "type", None) or getattr(cfg, "expectation_type", None)
    return {"expectation": str(etype or type(er).__name__),
            "column": column,
            "unexpected_count": (er.result or {}).get("unexpected_count") if isinstance(er.result, dict) else None}


def _summarize(result, score_threshold: float) -> dict[str, Any]:
    runs = successes = 0
    failures: list[dict[str, Any]] = []
    for vr in _iter_suite_results(result):
        for er in vr.results:
            runs += 1
            successes += int(bool(er.success))
            if not er.success:
                failures.append(_er_info(er))
    score = successes / runs if runs else 0.0
    return {"expectations_run": runs, "expectations_succeeded": successes,
            "quality_score": score, "passed": bool(score >= score_threshold) and bool(result.success),
            "failures": failures[:20]}


def validate_dataframe(context, dataframe: pd.DataFrame, *, kind: str = "batch",
                       score_threshold: float = 0.999) -> dict[str, Any]:
    """Прогнать чекпойнт GE по датафрейму; вернуть сводку и (для kind=batch) список
    непрошедших ожиданий. Бросать ничего не должно — гейт принимает решение по passed."""
    batch_def = _ensure_datasource(context)
    if kind == "batch":
        suite = _ensure_suite(context, "fraud_batch_raw", batch_expectations())
        cp = _ensure_checkpoint(context, "batch_checkpoint", batch_def, suite, with_docs=False)
    elif kind == "marts":
        suite = _ensure_suite(context, "fraud_marts_quality", marts_expectations())
        cp = _ensure_checkpoint(context, "marts_checkpoint", batch_def, suite, with_docs=True)
    else:
        raise ValueError(kind)
    result = cp.run(batch_parameters={"dataframe": dataframe})
    return _summarize(result, score_threshold)


def build_data_docs(context) -> str:
    context.build_data_docs()
    site = Path(GE_ROOT) / "uncommitted" / "data_docs" / "local_site"
    if not site.exists():
        site = Path(GE_ROOT) / "expectations"
    return str(site)
