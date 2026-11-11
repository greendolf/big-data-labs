# Лабораторная работа 2 — вариант 4

Сравнение инструментов обработки данных (Pandas, Dask, Spark) на реальном Big Data-кейсе из
[ЛР1](../lab1/README.md). Отчёт с измерениями, разбором планов, skew-экспериментом,
оптимизациями и рекомендациями — в [Jupyter Notebook](lab2_engines_benchmark.ipynb)
(структура повторяет части задания, выводы каждой секции считаются из замеров).

## Статус

| Компонент | Состояние |
|---|---|
| Код подготовки, бенчмарков, Spark-конвейера; юнит-тесты | **готово** (15/15 тестов OK) |
| Подготовка данных: дедуп, GE-контроль, Parquet, срезы | **выполнено** (числа ниже) |
| Измерительный прогон ноутбука (части 2–7) | **не выполнен** — финальные числа появятся после прогона |
| GPU-трек B/C | не запускался, причины — [gpu/README.md](gpu/README.md) |

## Задача и контракт

- Датасет: [IoT dataset for Intrusion Detection Systems (IDS)](https://www.kaggle.com/datasets/azalhowaide/iot-dataset-for-intrusion-detection-systems-ids),
  файл `BoTNeTIoT-L01-v2.csv` — тот же источник, что в ЛР1 (SHA-256 проверяется по provenance-файлу).
- Задача обработки (задание, задача 1): **агрегация с группировкой, сложный ключ из 3 полей**:

  ```sql
  GROUP BY Attack, Attack_subType, Device_Name
  SELECT COUNT(*)            AS rows,
         AVG(MI_dir_L0.1_mean)   AS mean_MI_dir_mean,
         AVG(HH_L0.1_magnitude)  AS mean_HH_magnitude,
         MAX(H_L0.1_variance)    AS max_H_variance
  ```

  Один и тот же контракт выполняют Pandas, Dask, PySpark, Polars, DuckDB; каждый результат
  сверяется с Pandas (parity: точные счётчики, средние с rtol 1e-5).

## Методика и ограничения

- Единый бюджет CPU: всем инструментам 2 потока (`OMP/OPENBLAS/MKL/NUMEXPR/POLARS=2`,
  DuckDB `SET threads=2`, JVM `-XX:ActiveProcessorCount=2`, Spark `local[2]`).
- CPU-движки выполняются в изолированных subprocess-воркерах (чистый RSS, честный провал OOM
  как валидный результат); PySpark — медиана из 3 прогонов + warmup внутри общей сессии
  (время старта JVM в замеры не входит).
- Метрики: wall-time, CPU-секунды (дерево процессов), пик RSS (выборка 50–100 мс),
  байты и операции чтения/записи (Windows io_counters); Spark дополнительно — профиль стадий
  и задач из event log (shuffle-байты, перекос времени задач).
- Срезы 100 МБ/1 ГБ — точные байтовые префиксы очищенного файла (не синтетика); состав атак в
  префиксах смещён относительно полного файла — доли публикует таблица части 1.
- Файловый кэш ОС не сбрасывался: холодное чтение с NVMe не моделировалось.

## Фактические результаты подготовки данных (замерено)

- Источник: 7 062 606 строк × 27 полей, 1 678 174 220 байт;
  SHA-256 `e3cb8a91…494d` (совпал с provenance из ЛР1).
- Дедупликация полных строк (после trim): 621 659 избыточных повторов удалено → 6 440 947 строк.
  Пропусков нет ни в одном из 27 столбцов; рассогласований `label`↔`Attack` — 0.
- Great Expectations 1.24.1: 481 expectation-проверок по всем 6 440 947 строк; 455 успешны,
  unexpected — только предизвестные pcc-выбросы источника (`HH_L0.1_pcc` 2 100,
  `HpHp_L0.1_pcc` 1 206 значений; см. ЛР1). Отрицательный контроль (испорченная метка) отклонён.
  Отчёт: [results/quality_report.json](results/quality_report.json).
- Форматы: cleaned CSV 1 753 704 935 байт (больше источника — DuckDB пишет double полной
  точности), cleaned Parquet (zstd) 513 993 924 байт (−70.7% к cleaned CSV).
- Срезы: 100 МБ — 99 999 969 байт / 369 222 строки; 1 ГБ — 999 999 813 байт / 3 670 658 строк.

## Окружение и воспроизведение

Требуются Python 3.12 и JDK 17/21 для PySpark (на Windows — Spark 4.x + JDK 21).

```powershell
# 1. Окружение
uv venv data\env-lab2 --python 3.12          # или py -3.12 -m venv data\env-lab2
data\env-lab2\Scripts\python.exe -m pip install -r requirements-lab2.lock.txt

# 2. Подготовка данных (дедуп, GE, Parquet, срезы) — нужна один раз
data\env-lab2\Scripts\python.exe lab2\prepare_ids.py --force

# 3. Юнит-тесты конвейера (без полного прогона)
data\env-lab2\Scripts\python.exe -m unittest discover -s lab2 -p "test_*.py"

# 4. Полный измерительный прогон ноутбука (часы, пик RAM до ~25 ГБ)
data\env-lab2\Scripts\python.exe lab2\run_notebook.py
```

Windows-специфика Spark (настроено автоматически `run_notebook.py`, для ручных прогонов — вручную):

- `JAVA_HOME` может «висеть» на несуществующем пути; runner берёт `JAVA_HOME`, иначе
  `data/tools/lab2_java_home.txt` (путь к JDK), иначе ищет JDK 17/21 в реестре.
- Hadoop 3.x на Windows требует `winutils.exe` (event log/права каталогов): положить
  `winutils.exe` (из [cdarlint/winutils](https://github.com/cdarlint/winutils), проверено с
  hadoop-3.3.6) в `data/tools/hadoop/bin/winutils.exe`; runner выставит `HADOOP_HOME`.

Runner ограничивает библиотеки 2 потоками, держит Spark local-dirs и event log в игнорируемом
`data/lab2/`, сохраняет failed-исполнение в `data/lab2/failed_execution.ipynb` при ошибке и
требует ≥5 inline-графиков в успешном исполнении.

## Структура

| Файл | Назначение |
|---|---|
| [lab2_engines_benchmark.ipynb](lab2_engines_benchmark.ipynb) | Отчёт: части 1–7, замеры, выводы, автопроверки |
| [prepare_ids.py](prepare_ids.py) | Часть 1: источник→чистка→GE→Parquet→срезы |
| [benchmark_engines.py](benchmark_engines.py) | Части 2/6: subprocess-бенчмарки pandas/dask/polars/duckdb, проверка «Pandas vs полный датасет» |
| [spark_pipeline.py](spark_pipeline.py) | Части 2–5: сессия/замеры/plan/event log/skew/оптимизации |
| [run_notebook.py](run_notebook.py) | Воспроизводимый прогон ноутбука (env, JDK, kernel) |
| `test_*.py` | 15 юнит-тестов: срезы, качество, parity движков, парсер event log, skew-статистика, Spark vs Pandas |
| [gpu/README.md](gpu/README.md) | Часть 6: треки B/C — что выполнено и почему |
| [requirements_checklist.md](requirements_checklist.md) | Каждый подпункт задания → секция/артефакт |
| `results/` | Артефакты измерений (CSV/JSON/PNG/TXT) |

Данные, venv, временные файлы Spark остаются в игнорируемом `data/`. Оцениваемые результаты
ноутбука появятся в `results/` после измерительного прогона (см. Статус).
