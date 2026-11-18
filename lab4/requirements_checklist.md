# ЛР4 — карта соответствия пунктам задания

Все статусы — после полного прогона ноутбука (`NB_EXIT=0`, 34 ячейки / 19 кодовых, PNG
встроенные, `## METRICS ##`) и зелёных юнит-тестов (26/26) + CI-smoke (6/6 за 12.5 c).
✔ — выполнено, артефакт на месте; числа — из фактического прогона.

## Оценивание (6 критериев задания)

| Критерий | Балл | Где доказательство | Статус |
|---|---|---|---|
| 1. MLflow интеграция | 2.0 | 4 эксперимента-run с params/metrics/artifact (ноутбук ч.1–3), Model Registry `FraudModel` (v1 `@production`, v2 `@candidate @canary`), гейт AUC≥0.85 + latency, сравнение версий `batch-vs-online` run; скрины `results/mlflow_*.png` | ✔ |
| 2. Feature Store | 2.0 | `feast_repo/` + `features.py`; `FEATURE_MATCH_TEST_PASSED=True`, max_abs_diff 1.06e-04 (ноутбук ч.2, smoke шаг 1, тест в CI) | ✔ |
| 3. Инференс в production | 1.5 | FastAPI `/predict` с Feast online; p50/p95/p99 35.9/37.9/38.8 мс, 27.2 rps, 0 ошибок (600 req); canary 5% + shadow + promote/rollback (`deploy_experiment`, ноутбук ч.4) | ✔ |
| 4. Мониторинг данных и моделей | 1.5 | PSI/KS + Evidently (html/json в `results/` и в MLflow), дрейф-фикстура 0.3478 → алерт → retrain; чистое окно 0.0 (автопроверка); ноутбук ч.5 | ✔ |
| 5. Batch vs online | 1.5 | River prequential vs замороженная batch, скользящий AUC по окнам, `ONLINE_BATCH_DIFFERENCE=-0.0028`, рекомендации по стратегии (ноутбук ч.6) | ✔ |
| 6. Отчётность и документация | 1.5 | ноутбук по частям задания с графиками/выводами, README с METRICS и оговорками, чек-лист, воспроизводимость (`build_notebook.py` → `run_notebook.py`) | ✔ |

## Часть 1. Прототипирование и выбор модели

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Тип задачи, EDA (распределения, корреляции, проблемы) | ноутбук ч.0–1: классификация 8% позитивов, распределения фич атак, пропуски/шкалы | ✔ |
| 3–5 моделей, сравнение (AUC/F1/PR) | 4 модели (logreg, random_forest, xgboost, hist_gb): таблица AUC/avg_precision/f1/train_sec/infer_ms | ✔ |
| Базовый пайплайн (imputer/scaler/classifier) | пайплайны внутри `compare_models`; для DP-ветки scaler явным шагом | ✔ |
| MLflow-логирование эксперимента | каждый кандидат — run с params/metrics/артефактом модели (cloudpickle) | ✔ |
| Обоснование выбора + скриншот MLflow UI | random_forest (равный AUC, выше avg_precision), `results/mlflow_runs.png`, аудит AUC=1.0 в ноутбуке | ✔ |

## Часть 2. Feature Store

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Какие фичи offline/online, что критично | ноутбук ч.2: 23 фичи события + 7 агрегатов Feast; критичны все агрегированные | ✔ |
| Feast-репозиторий (`features.py`, entity, FeatureView) | `feast_repo/features.py`: entity `device`, `device_stats` (7 фич, ttl=2 дня) | ✔ |
| historical vs online и согласованность | `get_historical_features` vs `get_online_features`, assert допуск 1e-3: **1.06e-04** | ✔ |
| Вывод про feature mismatch | ноутбук ч.2 (одна дефиниция SQL → расхождение только сериализации) | ✔ |
| Без утечек в фичах | агрегаты по label найдены и удалены в аудите; фиксируется smoke/тестами состава | ✔ |

## Часть 3. Интеграция с MLflow

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Tracking server (sqlite+artifacts) | `mlflow server --backend-store-uri sqlite:///data/lab4/mlflow.db` (команда в README) | ✔ |
| Жизненный цикл: обучение→registry→production | `register_production` (гейт) → v1 → алиас `production` (MLflow 3.x: aliases вместо stages) | ✔ |
| Сравнение версий разных экспериментов + логирование difference | run `batch-vs-online`: `batch_auc`, `online_auc`, `difference` (ноутбук ч.6) | ✔ |
| Валидация перед деплоем (AUC≥0.85, latency) | `validate_for_deploy`: auc=1.0, batch1000=27.3 мс → passed (ноутбук ч.3, скрины реестра) | ✔ |

## Часть 4. Инференс в production

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| REST + Feature Store + ошибки | FastAPI `/predict`: payload 23 фич + Feast online; 422 битый JSON, 503 при отказе стора, `cold_start` без истории | ✔ |
| Метрики p50/p95/p99, throughput, ошибки | 600 сквозных запросов: 35.9/37.9/38.8 мс, 27.2 rps, error_rate 0; INFERENCE_LATENCY_MS=37.94 < 50 | ✔ |
| Canary 5%, shadow, автооткат | `deploy_experiment`: 2 516 req канарейке, AUC-контроль, shadow mean\|Δ\| 0.0004 / agreement 99.99%, promote v2@canary; rollback-ветка (алиасы остаются, канарейка гасится) реализована, на стенде сработала бы при ΔAUC < −0.02 | ✔ |
| Граничные значения/нагрузка | негативные запросы (422), cold_start без истории устройства, max запроса 55 мс, зонд 600 req | ✔ |

## Часть 5. Мониторинг

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| PSI, KS-тест | `monitor.py`: свой PSI (10 бинов) + KS по фичам | ✔ |
| Evidently + логирование в MLflow | `evidently_drift.html/json` в `results/`; run `drift-monitor` с `data_drift_score` (скрин) | ✔ |
| Пропуски/диапазоны | фикстура и скоринг окон по 10 фичам семейства; NaN-гарды в сервинге | ✔ |
| Пороги, алерты | drift_score>0.3 → алерт (фикстура 0.3478 сработала; чистое окно 0.0 — нет); падение AUC>0.15 → алерт; алерт-лог `data/lab4/alerts.log` (в проде — webhook) | ✔ |
| Авто-retrain (drift>0.3, acc>0.15, неделя) | `check_retrain_needed` — три правила; в прогоне retrain вызван по дрейфу, модель-кандидат помечена на canary | ✔ |

## Часть 6. Batch vs online

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Пакетное обучение + registry + canary | prod-путь ч.1–4 (batch random_forest → v1 → canary-воронка) | ✔ |
| Онлайн-обучение (River, learn_one, логирование) | `online.py`: River Pipeline(StandardScaler→LogisticRegression, SGD 0.05) prequential `predict_one/learn_one`, скользящий AUC по окнам 10k/2.5k, run в MLflow | ✔ |
| Сравнение: точность/latency/сложность/cost | ноутбук ч.6: AUC-окна до/после дрейфа, таблица latency batch/online, qualitative cost | ✔ |
| Рекомендации (когда batch/online/гибрид) | выводы ч.6: гибрид batch-период + online между релизами | ✔ |

## Часть 7. Уникальное требование (DP, вариант 4)

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Реализация DP-обучения | `dp.py`: DP-SGD (L2-clip по 1.0, гауссов шум, bias как вес фиктивной фичи), RDP-accountant полной композиции | ✔ |
| Измерение эффекта | кривая AUC(ε) на 6 точках (∞…4): baseline 0.9993 → ε*=4.0: 0.9986, **UNIQUE_REQUIREMENT_IMPACT=−0.0007** | ✔ |
| Сравнение с baseline | ε=∞ строка той же кривой + линия допуска −0.02 на графике | ✔ |
| Вывод применимости | ноутбук ч.7 + README: граница ε_min=2.54 без sampling amplification; малая цена — следствие сверхразделимости данных, указано явно | ✔ |
| Требования сценария: latency<50 / FPR<1% / late data | p95 37.2 мс; FPR 0.009% (`fpr_check.py`); Feast as-of join корректен при переупорядоченном приходе, обсуждение поздней разметки — README/ноутбук | ✔ |

## Часть 8. Прогноз сенсора и автоотключение

| Требование | Реализация/доказательство | Статус |
|---|---|---|
| Данные NAB, 2 аварии, вторая — holdout | `part8.py` INCIDENTS; удержание проверено: пороги тюнингованы только по accident_1 | ✔ |
| Рекурсивная vs прямая + квантиль | Theta (StatsForecast) рекурсивно + HistGB с лагами прямо; q0.9 = med+1.2816·σ·√h (модельная оговорка) | ✔ |
| Метрики pinball/MASE | pinball q0.9 = 0.501; MASE(h24) = 1.959 | ✔ |
| Конъюнкция 4 условий + обоснование цен | P>p\*=0.3, CUSUM(0.5/4), persistence 1, cooldown 24; dev=4 °F; обоснование ценой ошибки в ноутбуке | ✔ |
| Lead/precision/recall/ложные/пропущенные, без point-adjust | visible: lead 18 ч, 0.16 fp/день; holdout: обнаружена, lead 12 ч, 0.12 fp/день; пропущенных нет | ✔ |
| Рост ошибки по горизонтам | кривая MAE рекурсивной vs прямой (ноутбук, PNG) | ✔ |
| График с квантильной полосой, журнал срабатываний, вывод о цене ошибки | PNG ч.8 + print журнала + вывод (пропуск дороже ложного) | ✔ |

## Оформление

| Требование | Реализация | Статус |
|---|---|---|
| Вариант B: Jupyter Notebook с нуля воспроизводимый | `build_notebook.py` → `run_notebook.py`; NB_EXIT=0, авто-прогон PNG-стеток | ✔ |
| Структура по разделам, комментарии, визуализация, выводы по разделам | 34 ячейки по частям; ≥4 PNG; «Выводы по разделам» | ✔ |
| README (вариант/сценарий/запуск/выводы) | `README.md` | ✔ |
| Автостетки + фиксированные METRICS | 3 assert из задания + smoke; 5 метрик `## METRICS ##` — все пропечатаны | ✔ |
