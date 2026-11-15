# Лабораторная работа 3 — вариант 4

[![Lab3 CI](https://github.com/greendolf/big-data-labs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/greendolf/big-data-labs/actions/workflows/ci.yml)

**Вариант 4 — IoT Sensor Data (датасет BoTNeTIoT-L01-v2), сценарий 3 — Fraud Detection.**

Промышленный ETL/ELT-конвейер: микро-батчи → контракт схемы → построчная валидация с DLQ →
идемпотентная загрузка (`MERGE` по `idempotency_key`) → dbt (staging→marts, тесты, SCD2) →
витрина риск-скоринга устройств → оркестрация Prefect → GE-гейты → мониторинг, канарейка, CI.
Отчёт с измерениями и автопроверками — [Jupyter Notebook](lab3_fraud_pipeline.ipynb)
(структура повторяет части задания; метрики считаются из данных прогона).

Почему Fraud Detection: в IoT-датасете атаки устройств — естественный аналог мошенничества
(устройство = «клиент», событие = «транзакция», атака = «фрод»). Остальные сценарии
потребовали бы выдумывать смысл колонок или отсутствующие данные.

## Статус

| Компонент | Состояние |
|---|---|
| Модули пайплайна + юнит-тесты | **готово** (15/15 OK; `python -m unittest discover -s lab3 -p "test_*.py"`) |
| CI (`.github/workflows/ci.yml`) | **готово**: установка по локу `requirements-lab3.linux.lock.txt`, юнит-тесты, `python -m lab3.smoke` (e2e на синтетике, зелёный локально) |
| Полный прогон ноутбука | **выполнен**: 20 кодовых ячеек, 6 встроенных PNG, автостетки зелёные, `## METRICS ##` ниже |
| Prefect UI-скриншоты | `results/prefect_ui_flow_runs.png`, `results/prefect_ui_flow_run_gantt.png` (retry-логи видны) |

## Стек и обоснование замены

| Требование задания | Исполнено здесь | Продуктовый аналог |
|---|---|---|
| DWH с ACID/UPSERT | DuckDB 2.0, `MERGE INTO` по md5-ключу | Delta Lake / Iceberg на Spark |
| Time Travel / rollback | версионные снапшоты `snapshots.*` + `dbt snapshot` (SCD2) | Delta `VERSION AS OF` / `RESTORE` |
| Оркестрация | Prefect 3 (retries с backoff, dynamic tasks, UI на :4200) | Airflow (на Windows не ставился — Prefect равноценно закрывает пункт) |
| Качество данных | Great Expectations 1.25: сьюты, чекпойнты, Data Docs | — |
| Трансформации | dbt-duckdb: incremental-модели, tests, snapshots, vars, docs | dbt на любом DWH |

Методологические рамки: **event_time — replay-модель** (в IDS-датасете нет временных меток,
зафиксировано в ЛР1; синтез детерминирован, `batching.py`), **SLA 50 мс проверяется честно**
на micro-batch эмуляции и не выполняется — см. «SLA» ниже.

## Быстрый старт

```powershell
# 1. данные: сначала выполняется ЛР2 (создаёт data/lab2/inputs/ids_cleaned.parquet)
# 2. окружение
uv venv data\env-lab3 --python 3.12
uv pip install --python data\env-lab3\Scripts\python.exe -r requirements-lab3.txt
# 3. тесты и smoke
data\env-lab3\Scripts\python.exe -m unittest discover -s lab3 -p "test_*.py"
data\env-lab3\Scripts\python.exe -m lab3.smoke
# 4. полный исполненный отчёт
data\env-lab3\Scripts\python.exe lab3\run_notebook.py
```

## Структура

```
lab3/
  warehouse.py       # DuckDB-хранилище: raw/ops/snapshots, MERGE, снапшоты, DLQ, аудит
  batching.py        # микро-батчи + replay-модель event_time + фикстуры (dirty/drift/late)
  contract.py        # контракт схемы (алиас-fallback) и построчные правила → DLQ
  pipeline.py        # процесс батча: extract(retries-сырьё) → контракт → DLQ → MERGE
  prefect_flow.py    # поток fraud-nightly: retries+backoff, dynamic map алертов
  quality.py         # GE: персистентный контекст, сьюты batch/marts, Data Docs
  dbt_runner.py      # dbt build/snapshot/docs как субпроцесс с парсингом run_results
  canary.py          # канарейка ELT (dbt-target canary + var), smoke-гейт, промоушен, откат
  monitoring.py      # графики из ops.audit_runs, SLA-вердикт (p50/p95)
  dbt_graph.py       # рендер DAG из dbt manifest.json
  smoke.py           # e2e-проверка для CI на синтетике (без реальных данных)
  test_*.py          # 15 юнит-тестов
  dbt/               # dbt-проект (staging, marts, snapshots, macros, profiles)
  great_expectations/ # GE контекст (сьюты/чекпойнты на диске)
  results/           # PNG-графики, скриншоты UI, JSON-сводки
data/lab3/           # рантайм (warehouse.duckdb, батчи, prefect home) — в .gitignore
.github/workflows/ci.yml
```

## Фактические результаты (замерено на полном прогоне; см. METRICS в ноутбуке)

- Источник: 6 440 947 строк (выход ЛР2), 20 микро-батчей ~319k строк; сыро загружено
  **6 440 923** строки (24 в DLQ).
- **Идемпотентность**: переигрывание батча b003 → **0 вставок / 318 848 обновлений**
  (`IDEMPOTENCY_TEST_PASSED: True`); автопроверка в ноутбуке и юнит-тестах.
- **DLQ**: 24 строки фикстуры (пустые ключи / `label=7` / отрицательная дисперсия) —
  маршрутизированы с причиной, пайплайн не упал; отрицательная ковариация — законное
  значение (проверено по min() источника), в DLQ не попадает — закреплено тестом.
- **Schema drift**: батч b012 с переименованной колонкой восстановлен алиас-мапой
  (fallback на предыдущую схему), неизвестная колонка отброшена с алертом; неизлечимый
  дрейф роняет батч в `failed` (тест).
- **Late-arriving**: ~49.5k событий (≈1%) пришли после watermark (max event_time − 4 ч),
  загружены с `is_late=true`, догнали витрины через incremental-модели — ни одно не потеряно.
- **Time Travel**: откат `raw.events` к снапшоту 5-го батча точен до строки (автопроверка);
  dbt snapshot даёт SCD2-историю профилей.
- **GE**: батч-гейт 22 ожидания × прогон, score 1.0; витринный гейт score 1.0;
  негативный контроль (испорченный label) валится — GE не «всегда зелёный»; Data Docs
  собираются в GE-контексте.
- **dbt**: 3 модели + snapshot, тесты unique/not_null/accepted_values/relationships —
  все pass; граф DAG в `results/dbt_dag.png`.
- **Канарейка**: dbt-target `canary` с `risk_severity_weight=0.6` — separate схемы,
  GE smoke-гейт, промоушен с пред-снапшотом и точный откат (проверено).
- **CI**: `lab3.smoke` — e2e на 6 000 синтетических строк: загрузка, DLQ=8, replay 0
  вставок, dbt build pass.

Блок `## METRICS ##` из исполненного ноутбука (значение `PIPELINE_DURATION_SEC` — полное
время прогона, включая загрузку всех 20 батчей и GE на 319k строк):

```text
IDEMPOTENCY_TEST_PASSED=True
DLQ_ERROR_COUNT=24
DATA_QUALITY_SCORE=1.00
PIPELINE_DURATION_SEC=927.30
BATCHES_PROCESSED=20
ROWS_RAW=6440923
ROWS_LATE=368339
SLA_50MS_PASSED=False
SLA_P95_MS_PER_EVENT=34039.5
```

`ROWS_LATE` включает переигранный батч b003 (тест идемпотентности пришёл «после водопада» —
корректно помечен late) плюс фикстуру поздних данных b019/b020 (49 491) — см. пояснение
в ноутбуке.

## SLA 50 мс — честно про невыполнение

Задание требует latency ≤ 50 мс. Это **стриминговое** требование; данный стек — пакетная
(micro-batch) обработка, и в такой модели задержка события = время обработки его батча:

| Метрика (замерено) | Значение |
|---|---|
| p50 задержки события (время батча) | ≈ 20.8 с |
| p95 задержки события | ≈ 34.0 с |
| Амортифицированная стоимость на событие (пропускная способность) | **0.068 мс** |

Вывод: по пропускной способности стек более чем достаточен, по задержке события —
**SLA не выполняется, и не может выполняться** классом системы. Для реальных 50 мс нужны
Kafka + Flink (или Spark Structured Streaming с обработкой единичных событий/микро-окон)
и low-latency upsert-хранилище. Реализовывать это на учебном стенде — значит мухлевать
с цифрами; вместо этого факт измерен, вердикт в ноутбуке — `SLA_50MS_PASSED: False`.

## Как ложится на Delta/Databricks (справочно)

- `MERGE INTO raw.events` ↔ Delta `MERGE` (то же SQL-ядро);
- снапшоты ↔ `RESTORE TABLE ... VERSION AS OF` (дешёвый time travel вместо копий);
- `is_late`/watermark ↔ Structured Streaming watermark;
- dbt-duckdb ↔ dbt-spark без изменения моделей (кроме типов-агностик SQL).
