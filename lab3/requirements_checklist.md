# ЛР3 — карта соответствия пунктам задания

Все статусы — после полного прогона ноутбука (`RUNNER_EXIT=0`, 20 кодовых ячеек,
6 встроенных PNG) и зелёных юнит-тестах (15/15) + CI-smoke. ✔ — выполнено, артефакт на месте.

## Оценивание (6 критериев задания)

| Критерий | Балл | Где доказательство | Статус |
|---|---|---|---|
| 1. Идемпотентность | 2.0 | `warehouse.upsert_raw` (MERGE по md5-ключу), ноутбук ч.2: replay b003 → 0 вставок/318 848 обновлений; тесты `test_pipeline.test_replay_noop` | ✔ |
| 2. Обработка ошибок | 2.0 | retries+backoff в `prefect_flow` (flaky-экстрактор в ноутбуке ч.6), DLQ 24 строки с причинами (ноутбук ч.2, тесты), schema drift b012: алиас-fallback + алерт (`contract.py`, `test_contract.py`) | ✔ |
| 3. Оркестрация | 2.0 | `prefect_flow.py`: flow fraud-batch/fraud-nightly, зависимости задач, dynamic map алертов; скриншоты UI `results/prefect_ui_*.png` | ✔ |
| 4. Тестирование данных (GE) | 1.5 | `quality.py` + GE-контекст: сьюты batch (22 ожидания) и marts (5), чекпойнты, гейты до/после трансформаций, негативный контроль, Data Docs | ✔ |
| 5. DBT-трансформации | 1.5 | `lab3/dbt/`: staging→marts incremental, тесты (unique/not_null/accepted_values/relationships), snapshot SCD2, макрос, vars, docs + граф `results/dbt_dag.png` | ✔ |
| 6. Отчётность и мониторинг | 1.0 | `monitoring.py`: аудит-таблица `ops.audit_runs`, 3 графика `results/`, SLA-вердикт; ноутбук с METRICS и выводами | ✔ |

## Часть 1. Проектирование

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| ETL vs ELT с обоснованием | Ноутбук ч.1: гибрид (лёгкий ETL на входе + ELT в dbt), объём/инфраструктура/частота изменений | ✔ |
| Схема архитектуры (диаграмма) | Ноутбук ч.1 (блок-схема источников→извлечение→трансформация→хранилище→аналитика с точками оркестратора) | ✔ |
| Матрица требований | Ноутбук ч.1: таблица требование/решение/обоснование | ✔ |
| Выбор технологий + сравнение затрат | Ноутбук ч.1 + README: Prefect vs Airflow (Windows, инфраструктура), DuckDB vs Databricks (цена/масштаб), dbt/GE OSS | ✔ |
| Точки интеграции оркестратора, обработка ошибок в проекте | `prefect_flow.py` (задачи = точки интеграции), DLQ/retries/watermark на схеме | ✔ |

## Часть 2. Идемпотентность

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Идемпотентный ключ на извлечении | `make_idempotency_key` (md5 27 колонок + event_time) | ✔ |
| UPSERT вместо INSERT | `MERGE INTO raw.events` (MATCHED UPDATE / NOT MATCHED INSERT) | ✔ |
| Двойной прогон: объём не вырос, дублей нет | Ноутбук ч.2: replay → `inserted=0`, assert; `test_replay_noop`; уникальность PK проверена | ✔ |
| Time Travel / откат | `snapshot_raw`/`rollback_raw` (версии `snapshots.raw_events_*`); демо отката к b005 в ноутбуке — точное совпадение строк; Delta-аналог `VERSION AS OF` описан | ✔ |
| Проверка в тестах | `test_pipeline` (полный цикл + replay) | ✔ |

## Часть 3. Обработка ошибок

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Retries с экспоненциальной задержкой, лимит попыток | Prefect task `retries=2, retry_delay_seconds=[1,2,4]`; flaky-извлечение в ноутбуке ч.6 (1-я попытка падает, поток зелёный) | ✔ |
| Логирование ошибок | `ops.audit_runs` (стадия/статус/метрики на каждый шаг), `ops.dlq`, console | ✔ |
| DLQ: хранилище, маршрутизация, алерты | `ops.dlq` (24 строки), `split_clean_and_bad` + `alert_hook` при превышении порога; примеры записей в ноутбуке ч.2 | ✔ |
| Schema drift: проверка, fallback, алерты | `contract.apply_schema_contract`: обязательные колонки, алиас-мапа (предыдущая схема), drop неизвестных + алерт, `SchemaDriftError` при потере обязательных; батч b012 в ноутбуке | ✔ |
| Искусственный сбой на каждом этапе, восстановление | Фикстуры b007 (грязные строки) / b012 (дрейф) / flaky-extract; регресс-тест на ковариацию (ложных DLQ нет); тесты провала обязательной колонки | ✔ |

## Часть 4. Оркестрация

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Поток с задачами extract→validate→transform→load | `flow_fraud_batch`, `flow_fraud_nightly` (батчи → dbt → GE-гейт → алерты) | ✔ |
| Зависимости, динамические задачи | последовательные зависимости + `task_risk_alert.map` по «горячим» устройствам (количество задач зависит от данных) | ✔ |
| Мониторинг: логирование, метрики, алерты | аудит-таблица + `monitoring.py` + risk-алерты Prefect | ✔ |
| Проверка вручную + UI оркестратора | прогон flow в ноутбуке; скриншоты Prefect UI (`results/prefect_ui_flow_runs.png`, `..._gantt.png`) | ✔ |

## Часть 5. Great Expectations

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Настроенный контекст | `lab3/great_expectations/` (file-context, коммитится) | ✔ |
| Сьюты: пропуски/диапазоны/уникальность/связи | batch-сьют 22 ожидания (not_null, диапазоны, unique ключа, домены label/Attack), marts-сьют + relationships в dbt-тестах | ✔ |
| Интеграция в пайплайн (гейт) | `validate_dataframe` до загрузки; провал → батч не грузится; негативный контроль в ноутбуке | ✔ |
| Регулярность: до загрузки, после трансформаций, задача оркестратора | гейт batch (per-batch), гейт marts (после dbt, в `flow_fraud_nightly` задачей) | ✔ |
| HTML-отчёты | Data Docs (`quality.build_data_docs`) | ✔ |

## Часть 6. DBT

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Структура проекта (models/staging, marts, macros, tests) | `lab3/dbt/` | ✔ |
| Staging-очистка + marts-бизнес-логика | `stg_events` (incremental), `fct_attack_events`, `dim_device_profile` (риск-скоринг) | ✔ |
| Макросы | `attack_severity` | ✔ |
| Тесты в schema.yml + результаты | unique/not_null/accepted_values/relationships; `dbt build` — все pass (ноутбук ч.4) | ✔ |
| Снапшот/история | `device_profile_history` (check-strategy SCD2) | ✔ |
| DBT Docs + граф зависимостей | `docs generate` + `results/dbt_dag.png` | ✔ |
| Интеграция в оркестратор | `task_dbt_build`/`task_dbt_snapshot` в Prefect-потоке | ✔ |

## Часть 7. CI/CD

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| GitHub Actions | `.github/workflows/ci.yml`: unit-тесты + `lab3.smoke` (e2e без реальных данных) на push по `lab3/**` | ✔ |
| GE и dbt в CI | smoke-контур: загрузка+DLQ+replay+`dbt build`; GE-гейты — в локальном контуре прогона (полный GE в CI не гоняется — время; оговорено) | ✔ |
| Canary release | `canary.py`: dbt-target canary + var `risk_severity_weight`, изолированные схемы | ✔ |
| Smoke-тесты перед деплоем | `smoke_gate` (GE по канареечной витрине) — промоушен только при passed | ✔ |
| Автоматический откат | `promote_canary` с пред-снапшотом + `rollback_canary` (демо в ноутбуке) | ✔ |
| Мониторинг после деплоя: сравнение метрик до/после, drift | стабильная vs канареечная витрина (средний risk до/после печатается), data-drift монитор `monitoring.py` | ✔ |

## Обязательные элементы оформления

| Требование | Реализация | Статус |
|---|---|---|
| Вариант B: воспроизводимый Notebook по разделам задания | `lab3_fraud_pipeline.ipynb` (35 ячеек, ч.1–7 + METRICS + выводы), `build_notebook.py` — источник ячеек, `run_notebook.py` — headless-исполнение | ✔ |
| Визуализация | 3 графика мониторинга + dbt DAG + скриншоты Prefect UI | ✔ |
| README.md (вариант, сценарий, запуск, выводы) | `lab3/README.md` | ✔ |
| Автоматические проверки идемпотентности и DLQ | assert'ы ноутбука + юнит-тесты (ровно паттерн задания: два прогона → счётчик не вырос; corrupt → DLQ>0) | ✔ |
| Метрики `## METRICS ##` | финальная ячейка ноутбука: IDEMPOTENCY_TEST_PASSED, DLQ_ERROR_COUNT, DATA_QUALITY_SCORE, PIPELINE_DURATION_SEC (+доп.) | ✔ |
| Выводы по каждому разделу | markdown-блоки «вывод» в конце частей + итоговый раздел | ✔ |
| Структура репозитория как в задании | README в корне лаборатории, `dbt/`, `great_expectations/`, `tests (lab3/test_*.py)`, `.github/workflows/ci.yml` | ✔ |
