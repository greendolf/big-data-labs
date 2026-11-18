# Лабораторная работа 4 — вариант 4

[![Lab4 CI](https://github.com/greendolf/big-data-labs/actions/workflows/ci-lab4.yml/badge.svg?branch=main)](https://github.com/greendolf/big-data-labs/actions/workflows/ci-lab4.yml)

**Вариант 4 (последняя цифра зачётки) — уникальное требование: дифференциальная приватность.
Бизнес-сценарий 1 — Fraud Detection.** BoTNetIoT-L01 из ЛР1–ЛР3 (устройство = «клиент»,
событие = «транзакция», атака = «фрод»); ML-матрица собирается из хранилища ЛР3.

Production-обвязка вокруг fraud-модели: **MLflow** (трекинг 4 моделей, Registry, гейты
AUC≥0.85 и latency перед деплоем, алиасы production/candidate/canary), **Feast**
(offline/online-согласованность с автопроверкой `np.allclose`-допуска), **FastAPI**-инференс
с замерами p50/p95/p99 + canary 5% + shadow + автооткат, **мониторинг** дрейфа (PSI/KS,
Evidently) с детерминированным retrain-триггером, **batch vs online** (River, prequential),
**DP-SGD** — цена приватности с консервативным RDP-учётом, и **часть 8** — квантильный прогноз
температуры NAB с логикой автоотключения (порог настроен по видимой аварии, вторая — holdout).
Отчёт — [исполненный Jupyter Notebook](lab4_ml_production.ipynb) (вариант B, структура
повторяет части задания; все числа — из замеров прогона).

## Статус

| Компонент | Состояние |
|---|---|
| Модули + юнит-тесты | **готово** (26/26 OK; `python -m unittest discover -s lab4 -p "test_*.py"`) |
| CI (`.github/workflows/ci-lab4.yml`) | **готово**: установка по локу `requirements-lab4.linux.lock.txt`, юнит-тесты, `python -m lab4.smoke` (e2e на синтетике: Feast, MLflow, FastAPI, мониторинг, DP, часть 8 — зелёный локально за 12.5 c) |
| Полный прогон ноутбука | **выполнен**: 34 ячейки (19 кодовых), 4+ встроенных PNG, автостетки зелёные, `## METRICS ##` ниже, время 552 c |
| Скриншоты MLflow UI | `results/mlflow_runs.png`, `mlflow_run_detail.png`, `mlflow_registry.png`, `mlflow_drift_monitor.png` |
| Evidently-отчёт дрейфа | `results/evidently_drift.html` (+ JSON, логируется и в MLflow-run `drift-monitor`) |
| FPR < 1% (требование сценария) | **измерено отдельно**: `python -m lab4.fpr_check` → **FPR = 0.009%** (20 FP / 229 880) |

## Стек и обоснование замены

| Требование задания | Исполнено здесь | Продуктовый аналог |
|---|---|---|
| MLflow Tracking/Registry | MLflow 3.18, local backend sqlite + file-artifacts, `mlflow server` для UI | Managed MLflow / Databricks |
| Feature Store | Feast 0.49: local file source (DuckDB-агрегации) + sqlite online store | Feast + Redis + Spark offline |
| REST-инференс | FastAPI + uvicorn (in-process TestClient в замерах), payload → Feast online → pyfunc | KServe / SageMaker endpoint |
| Модели (3–5 шт) | sklearn 1.9 (random_forest, hist_gb, logreg) + XGBoost 3.4 | — |
| Онлайн-обучение | River 0.23 (StandardScaler → LogisticRegression(SGD), prequential) | River / Flink ML |
| Мониторинг | PSI/KS собственные + Evidently 0.7.23 (DataDriftTable, html/json) | Evidently Service / WhyLabs |
| DP-SGD (уник. требование) | NumPy-реализация `lab4/dp.py` + RDP-accountant полной композиции | TF-Privacy / Opacus |
| Часть 8 | StatsForecast Theta (рекурсивная) + HistGradientBoosting с лагами (прямая) | Chronos-Bolt/Prophet |

Замены с историей: **LightGBM 4.7.0** падал с `access violation` в `LGBM_DatasetSetField`
на повторных Dataset в одном процессе стенда (падение воспроизводилось нестабильно,
лечилось только сменой пакета) — прямая стратегия части 8 пересобрана на равноклассном
sklearn HistGradientBoosting; пакет из зависимостей удалён.

## Методологические рамки (важно для честного чтения цифр)

1. **event_time — replay-модель ЛР3** (в IDS-данных нет реальных меток времени; синтез
   детерминирован, см. ЛР3). Точка as-of для Feast — из неё.
2. **Метки синтетического ряда почти детерминированы признаками** — отсюда AUC = 1.0.
   Это измерено и исследовано (см. блок «AUC = 1.0» ниже), а не спрятано: цифра
   настоящая, вывод о ней — тоже.
3. **DP-учёт консервативный**: полная RDP-композиция без усиления Пуассоновской выборки
   (его дают Opacus/TF-Privacy; пририсовать «красивые» ε без настоящего аккаунтера —
   фейковая гарантия). Граница честности: ε_min(2000 шагов, δ=1e-6) = **2.54**; сетка
   бюджета сознательно обрывается на 4.0.
4. **Дрейф** — фикстура (масштаб +155% по семейству MI/H/HH + смена знака starling-угла
   на 15% строк): в 24-часовом replay-окне естественного дрейфа нет, механизм мониторинга
   проверяется управляемым сдвигом; на чистом окне drift_score = 0.0 (автопроверка).
5. **Часть 8**: абсолютный T_crit в ряду фиктивен (ряд 57–86 °F), поэтому порог локальный
   `base30(t) + dev`; обе аварии — относительные аномалии; тюнинг строго по видимой аварии,
   вторая — недотрагиваемый holdout.

## Быстрый старт

```powershell
# 0. данные: выполнены ЛР1–ЛР3 (нужен data/lab3/warehouse.duckdb с raw.events)
# 1. окружение
uv venv data\env-lab4 --python 3.12
uv pip install --python data\env-lab4\Scripts\python.exe -r requirements-lab4.txt
# 2. тесты и smoke (без больших данных)
data\env-lab4\Scripts\python.exe -m unittest discover -s lab4 -p "test_*.py"
data\env-lab4\Scripts\python.exe -m lab4.smoke
# 3. полный исполненный отчёт (пересобирает ноутбук и гоняет headless, ~9 мин)
data\env-lab4\Scripts\python.exe lab4\build_notebook.py
data\env-lab4\Scripts\python.exe -m lab4.run_notebook
# 4. MLflow UI (артефакты прогона)
data\env-lab4\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///data/lab4/mlflow.db `
  --default-artifact-root file:///<abs>/data/lab4/artifacts --port 5000
# 5. замер FPR продакшн-модели (требование сценария < 1%)
data\env-lab4\Scripts\python.exe -m lab4.fpr_check
```

## Структура

```
lab4/
  data_prep.py        # ML-матрица из хранилища ЛР3, time-split, фикстура дрейфа
  features.py         # SQL-агрегаты устройств (DuckDB) -> feature table для Feast
  feast_repo/         # Feast-репозиторий (entity device, feature view device_stats, 7 фич)
  mlpipeline.py       # сравнение моделей в MLflow, гейт валидации, registry+алиасы
  wrap.py             # ProbaWrapper: pyfunc-предсказание вероятностями (не метками)
  serving.py          # FastAPI /predict (Feast online, cold_start, 422/503), latency-зонд,
                      # canary 5% + shadow + автооткат по AUC-канарейки
  monitor.py          # PSI/KS, Evidently-отчёт, retrain-правила (drift_score>0.3 | auc_drop>0.15 | неделя)
  online.py           # River prequential vs замороженная batch, скользящий AUC
  dp.py               # DP-SGD (покомпонентный clip+шум) + RDP-accountant полной композиции
  part8.py            # прогноз/квантили/тревога NAB (CUSUM, persistence, cooldown), метрики
  fpr_check.py        # воспроизводимый замер FPR продакшн-алиаса (< 1%)
  smoke.py            # e2e для CI на синтетике (6 шагов, без больших данных)
  build_notebook.py   # единственный источник ячеек ноутбука (правим тут, не в .ipynb)
  run_notebook.py     # headless-исполнение с автостеткой PNG и failed_execution.ipynb
  lab4_ml_production.ipynb  # ИСПОЛНЕННЫЙ отчёт ( outputs внутри )
  test_*.py           # 26 юнит-тестов (dp/monitor/part8)
  results/            # MLflow UI-скрины, evidently-отчёты
  data/               # NAB ambient_temperature_system_failure.csv (часть 8)
data/lab4/            # рантайм (mlflow.db, artifacts, events.parquet) — в .gitignore
.github/workflows/ci-lab4.yml
```

## Автоматические проверки (из задания)

| Проверка | Где | Результат прогона |
|---|---|---|
| `np.allclose` offline vs online | ноутбук ч.2, smoke шаг 1 | `FEATURE_MATCH_TEST_PASSED=True` (max_abs_diff **1.06e-04**, допуск 1e-3 — float32 материализации) |
| `assert drift_score < 0.3` на чистом окне | ноутбук ч.5/финал | drift_score чистого окна **0.0** — пройдено; фикстура дрейфа 0.3478 > 0.3 детектируется |
| `assert latency < 50` мс | ноутбук ч.4, smoke шаг 3 | сквозной p95 **37.94 мс** (FastAPI+Feast+pyfunc, 600 запросов) — пройдено |
| smoke в CI | `lab4.smoke` | 6/6 шагов, 12.5 c (Feast, MLflow registry, FastAPI p95 4.2 мс, мониторинг, DP, часть 8) |

## Фактические результаты (замерено; полный блок — `## METRICS ##` ноутбука)

```text
FEATURE_MATCH_TEST_PASSED=True
DATA_DRIFT_SCORE=0.3478
INFERENCE_LATENCY_MS=37.94
ONLINE_BATCH_DIFFERENCE=-0.0028
UNIQUE_REQUIREMENT_IMPACT=-0.0007
```

подробнее (там же, в ноутбуке и `data/lab4/metrics.json`):

| Область | Замер |
|---|---|
| Feast offline/online | max_abs_diff = 1.06e-04 (float32), все 7 агрегатов совпали |
| Сравнение моделей | random_forest / xgboost / hist_gb: AUC ≈ 1.0; logreg 0.649; best → **random_forest** (продакшн v1) |
| Гейт перед деплоем | AUC 1.0 ≥ 0.85, батч 1000 строк 27.3 мс ≤ 50 — **passed**, v1 → алиас `production` |
| Инференс (600 req) | p50 35.9 / p95 37.9 / p99 38.8 мс, 27.2 rps, 0 ошибок, max 55 мс |
| Canary 5% + shadow | 2 516 запросов кандидату (xgboost, v2), AUC-канарейки 1.0 ≥ прода; shadow mean\|Δ\| 0.0004, agreement 99.99% → **promoted** |
| Мониторинг | PSI/KS: чистое окно 0.0, фикстура 0.3478 → алерт → retrain сработал; AUC-падение 0.0000 (метрика насыщена — см. ниже), Evidently-отчёт сохранён |
| Batch vs online | замороженная batch 1.0000 vs River prequential 0.9972 (до дрейфа 0.9973 / после 0.9975) → ONLINE_BATCH_DIFFERENCE = **−0.0028** |
| DP (уник. требование) | ε_min(2000,δ=1e-6)=2.54; baseline AUC 0.9993; ε=4 → 0.9986; ε\*=4.0, impact **−0.0007** |
| Часть 8 (видимая авария) | lead **18 ч**, 3 тревоги в окне допуска, 36 ложных вне окна (0.16/день) |
| Часть 8 (holdout, тайна) | **обнаружена**, lead **12 ч**, 0.12 ложных/день — пороги по ней не тюнинговались |
| Часть 8 (прогноз) | pinball q0.9 = 0.501; MASE(h=24) = 1.959; кривая MAE рекурсивной vs прямой по горизонтам — в ноутбуке |
| FPR (требование <1%) | 0.009% (20 FP / 229 880) при пороге 0.5, prod-алиас |

## AUC = 1.0 — что это и почему это не баг пайплайна

Все деревовидные модели набирают AUC ≈ 1.0 на тестовом окне. Мы не стали делать вид,
что это «просто хорошая модель», и измерили причины (ячейка «Аудит качества» в ноутбуке):

1. **Прямой утечки метки нет**: максимальная univariate AUC — `HpHp_L0.1_radius` 0.907;
   ни одна фича не равна метке. Найденная в ходе аудита **реальная утечка** — Feast-агрегаты
   `AVG(label)` и счётчик подтипов атак — была найдена, **удалена** из feature view, весь
   прогон пересобран без неё (это же ловят smoke и юнит-тесты на составе фич).
2. **Дело не в запоминании устройств**: leave-one-device-out (модель обучается без
   устройства X, тестируется на X) даёт AUC ≈ 1.0 на всех трёх контрольных unseen-устройствах.
3. **Разделимость нелинейна и кумулятивна**: logreg по всем 30 фичам — 0.649; дерево
   глубины 2 → 0.854, глубины 3 → 0.881, глубины 10 → 0.998; гбдт — 1.0. Атаки в
   синтетическом ряду ЛР1–ЛР3 оставлены в признаках почти детерминированным сигнатурным
   отпечатком — свойство данных учебной цепочки, а не ошибка ЛР4.

Честные следствия, которые мы не прячем: гейт AUC ≥ 0.85 проходит тривиально; падение
AUC на дрейф-фикстуре ≈ 0 (детектирует PSI, не AUC); canary-ΔAUC и online-vs-batch дельты
малы; «цена приватности» на ε ≥ 4 — −0.0007. Все эти метрики при этом сосчитаны
по-настоящему, а пайплайн от данных не зависит: поменяй датасет на реальный — код
не изменится, цифры пересчитаются.

## DP — цена приватности по-честному

DP-SGD (клип по L2 + гауссов шум, bias — вес фиктивной фичи, общий клип) обучает
логистическую регрессию на 200k×30 за 2000 шагов; бюджет считается RDP-accountant'ом
**полной композиции** (без усиления выборок):

| ε (запрошенный) | ∞ (baseline) | 20 | 12 | 8 | 5 | 4 |
|---|---|---|---|---|---|---|
| AUC (тест) | 0.9993 | 0.9987 | 0.9987 | 0.9987 | 0.9986 | 0.9986 |

ε\* = 4.0 (потеря ≤ 0.02), **UNIQUE_REQUIREMENT_IMPACT = −0.0007**. Меньше ε_min = 2.54
бюджет при наших 2000 шагов без sampling amplification недостижим — это граница честности
реализации, она нарисована на графике и написана в ноутбуке; «красивое» ε=1.27 из
учебных примеров потребовало бы Opacus/TF-Privacy, врать без них мы не стали.
Почему потеря так мала — см. раздел про AUC=1.0: на почти разделимых данных шум σ=63
клипованных градиентов почти не ломает ранжирование. Это следствие данных, а не
«бесплатности DP»; на данных без сверхразделимости кривая покупается дороже —
механизм от этого не меняется.

## Часть 8 — кратко

Порог локальный (`медиана 30 суток + dev=4 °F`), тревога — конъюнкция P(T>T_crit за 24 ч) > 0.3,
CUSUM (k=0.5, h=4), persistence k=1, cooldown 24 ч; параметры выбраны сеткой по видимой
аварии (lead↑, ложные/день↓). Метрики — в таблице выше; вывод про цену ошибки: пропуск
аварии дороже ложного отключения, поэтому сетка и выбрана по максимуму lead при
< 0.2 ложных/день. Без point-adjustment.

## Как ложится на cloud-аналог (справочно)

- Feast local file source ↔ Feast + Spark/S3 offline, Redis online — код feature view тот же;
- sqlite MLflow backend ↔ Postgres backend + S3 артефакты (`mlflow server` без изменений);
- FastAPI TestClient ↔ тот же uvicorn под KServe/ECS; логика `deploy_experiment`
  (canary/shadow/rollback) переносится в продакшн-роутинг без правок;
- `train_logreg_dp` ↔ Opacus `DPSequential` + `PrivacyEngine` (аккаунтер с усилением —
  единственный функциональный шаг вверх).
