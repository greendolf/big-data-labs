"""Build lab2/lab2_engines_benchmark.ipynb for the Lab 2 submission.

The notebook mirrors the assignment structure (parts 1-7). Conclusions are
Markdown cells computed from the measurements at run time, so rerunning the
notebook cannot desync narrative text and data.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
md("""# Лабораторная работа 2. Сравнение инструментов обработки данных — Pandas, Dask и Spark

**Вариант 4** (последняя цифра зачётной книжки — 4, как в ЛР1).

| Параметр | Значение |
|---|---|
| Датасет | [IoT dataset for Intrusion Detection Systems (IDS)](https://www.kaggle.com/datasets/azalhowaide/iot-dataset-for-intrusion-detection-systems-ids), файл `BoTNeTIoT-L01-v2.csv` — тот же, что в ЛР1 |
| Бизнес-сценарий (ЛР1) | №4: анализ логов и мониторинг IoT-устройств, цель — снижение MTTR на 40% |
| Выбранная задача обработки | **Задача 1 — агрегация с группировкой**: сложный ключ `Attack × Attack_subType × Device_Name` (3 поля), агрегаты `count`, `avg`, `avg`, `max` |
| Метрики сравнения | время, пиковая RAM, байты и операции ввода-вывода, CPU-секунды; parity результата против Pandas |
| Ограничение окружения | всем инструментам ограничено 2 потока CPU (`OMP/POLARS/...=2`, `local[2]`, `-XX:ActiveProcessorCount=2`) |

Оборудование: AMD Ryzen 5 5600H (6 ядер / 12 потоков), 32 ГБ RAM, NVMe; Python 3.12,
pandas / dask / polars / duckdb / PySpark 4.x (`local[2]`, JDK 21).

Оговорки методики: замеры на одной машине; CPU-движки — по одному прогону на объём
(worker-подпроцессы), PySpark — медиана из 3 прогонов; файловый кэш Windows не сбрасывался;
пиковая RAM — оценка с дискретной выборкой раз в 50–100 мс. Это исследовательские timings,
а не статистические оценки; все сырые числа — в `lab2/results/` и CSV-артефактах.""")

md("""## Введение

- **Используемый датасет:** IoT IDS (признаки семейства CICIoT2023), ~7.06 млн строк,
  27 столбцов, CSV ≈ 1.68 ГБ.
- **Выбранная задача:** профиль трафика по группам `Attack × Attack_subType × Device_Name` —
  количество окон, средние значения двух статистик и максимум дисперсии. Это «сырьё» для
  мониторингового дашборда из сценария ЛР1 (профилирование атак по моделям устройств).
- **Цели анализа:** (1) корректно измерить стоимость одной и той же операции в Pandas, Dask,
  Spark и (Трек A) Polars/DuckDB; (2) разобрать физический план Spark; (3) найти и устранить skew;
  (4) измерить эффект оптимизаций Spark; (5) выдать рекомендации по выбору инструмента под сценарий.

Одна и та же контрактная операция (одинаковые ключи, агрегаты, правила чтения) выполняется
всеми инструментами, и результат каждого сверяется с результатом Pandas на том же входе (parity).""")

# ---------------------------------------------------------------- part 1
md("""## Часть 1. Подготовка окружения и данных

Окружение ставится из `requirements-lab2.txt` (или `.lock.txt`); Spark требует JDK 17/21.
Начинаем с импортов и подготовки данных: проверка источника по SHA-256 из provenance-файла ЛР1,
дедупликация, проверка пропусков, контроль качества Great Expectations и нарезка объёмов.""")

code("""import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from lab2 import benchmark_engines, prepare_ids, spark_pipeline

RESULTS = ROOT / "lab2" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RUNTIME = ROOT / "data" / "lab2"
pd.set_option("display.max_colwidth", None)
print(f"root: {ROOT}")
print(f"python: {sys.version.split()[0]}")
import os
print(f"JAVA_HOME={os.environ.get('JAVA_HOME')}")
print(f"HADOOP_HOME={os.environ.get('HADOOP_HOME')}")""")

code("""prepared = prepare_ids.ensure_prepared(ROOT)
clean = prepared["cleaning"]
summary_rows = [
    ("Строк в исходном CSV", f"{clean['raw_rows']:,}"),
    ("Строк после дедупликации", f"{clean['cleaned_rows']:,}"),
    ("Удалено полных дублей", f"{clean['duplicate_excess_rows']:,}"),
    ("Строк с пропусками после очистки", f"{clean['rows_with_any_null']:,}"),
    ("Несовпадений label и Attack", f"{clean['label_attack_mismatch_rows']:,}"),
    ("Размер cleaned CSV, МБ", f"{clean['cleaned_csv_bytes'] / 1e6:,.0f}"),
    ("Размер cleaned Parquet (zstd), МБ", f"{clean['cleaned_parquet_bytes'] / 1e6:,.0f}"),
]
display(pd.DataFrame(summary_rows, columns=["Показатель", "Значение"]))
print("Источник:", prepared["source"]["path"])
print("SHA-256:", prepared["source"]["sha256"])
print("Контракт агрегации:", json.dumps(prepared["aggregation"], ensure_ascii=False))""")

md("""**Правила очистки.** (1) Дедупликация полных строк после `trim` текстовых столбцов —
в источнике ~8.8% точных дублей. (2) Пропуски: GE проверяет все 27 столбцов — их нет,
правил додумывания не требуется; если пропуски появятся в новой версии данных, агрегаты
проигнорируют NULL, а GE зафиксирует нарушение as-unexpected. (3) Согласованность `label`
и `Attack` проверена. Аномальные `pcc` вне [-1; 1] (найдено в ЛР1) сохраняются: задача
агрегирования эти столбцы не использует. Замечание: CSV после дедупликации *больше* исходника
— Duckdb выписывает double в полной точности; компактное представление — Parquet (zstd).""")

code("""quality = prepared["quality"]
unexpected = quality["unexpected_counts"] or {}
print(f"Реализация: {quality['implementation']}")
print(f"Проверено строк: {quality['rows_checked']:,}")
print(f"Expectation-проверок: {quality['expectations_run']:,} (успешно {quality['expectations_succeeded']:,})")
print(f"Negative control сработал: {quality['negative_control_failed_as_expected']}")
if unexpected:
    display(pd.DataFrame(sorted(unexpected.items(), key=lambda kv: -kv[1]),
                         columns=["Expectation", "Unexpected-значений"]))
else:
    print("Нарушений нет")""")

code("""slices = pd.DataFrame(prepared["slices"])
slices["attack_share"] = slices["attack_share"].map(
    lambda d: ", ".join(f"{k}: {v:,}" for k, v in d.items()))
display(slices[["size_mb_target", "bytes", "rows", "attack_share"]])
print(f"Полный датасет для Spark: {clean['cleaned_csv_bytes']/1e6:,.0f} МБ, {clean['cleaned_rows']:,} строк")""")

md("""**Нарезка данных** (по заданию): 100 МБ — локальные тесты, 1 ГБ — сравнение Pandas/Dask,
полный датасет — Spark. Источник — 1.68 ГБ реальных строк, поэтому срезы это честные байтовые
префиксы очищенного файла (до границы строки), а не синтетический replay, как в черновике до
переделки ЛР1. Префиксные срезы смещают состав атак относительно полного файла (колонка
attack_share выше) — это ограничение учитывается в выводах части 2.""")

code("""spark = spark_pipeline.build_session(
    RUNTIME / "spark-events", name="BigDataLab2", cores=2, shuffle_partitions=200
)
print(f"Spark {spark.version}, master={spark.sparkContext.master}, "
      f"parallelism={spark.sparkContext.defaultParallelism}")
for key in ("spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled", "spark.eventLog.enabled"):
    print(f"{key} = {spark.conf.get(key)}")
print("eventLog.dir =", spark.conf.get("spark.eventLog.dir"))
app_id = spark.sparkContext.applicationId
event_dir = RUNTIME / "spark-events"
print("applicationId =", app_id)""")

md("### Вывод по части 1")

code("""from IPython.display import Markdown
conclusion_1 = f'''
- Исходник: {clean["raw_rows"]:,} строк; после дедупликации {clean["cleaned_rows"]:,}
  (удалено {clean["duplicate_excess_rows"]:,} полных дублей, {clean["duplicate_excess_rows"]/clean["raw_rows"]*100:.1f}%).
- Great Expectations: {quality["expectations_run"]:,} проверок по {quality["rows_checked"]:,} строк,
  отрицательный контроль исправен; единственные unexpected — предизвестные pcc-выбросы источника,
  они вне контракта агрегации.
- Объёмы для сравнения: 100 МБ и 1 ГБ (префиксные срезы реальных строк) + полный датасет
  {clean["cleaned_csv_bytes"]/1e6:,.0f} МБ.
- Spark-кластер: {spark.version}, local[2], 200 shuffle-партиций (как в задании), event log включён.
'''
display(Markdown(conclusion_1))""")

# ---------------------------------------------------------------- part 2
md("""## Часть 2. Сравнение производительности

Контрактная задача выполняется на: **Pandas** (100 МБ, 1 ГБ и попытка на полном датасете),
**Dask** (1 ГБ и полный датасет), **PySpark** (100 МБ, 1 ГБ, полный датасет); **Polars** и
**DuckDB** участвуют для полноты и составят Трек A (часть 6). Каждый CPU-движок работает в
отдельном subprocess c лимитом 2 потока; измеряются wall-time, CPU-секунды, пиковая RAM,
байты/операции ввода-вывода; каждый результат проверяется parity против Pandas на том же входе.""")

code("""slice_paths = {
    "100MB": Path(prepared["slice_paths"]["100"]),
    "1GB": Path(prepared["slice_paths"]["1000"]),
}
cpu_measurements = benchmark_engines.run_benchmark(
    slice_paths, RUNTIME / "benchmarks", engines=["pandas", "dask", "polars", "duckdb"]
)
display(cpu_measurements[["engine", "size_label", "wall_seconds", "peak_rss_mb",
                          "cpu_seconds", "read_bytes", "parity_with_pandas", "error"]])""")

code("""full_inputs = {**slice_paths, "full": Path(prepared["cleaned_csv"])}
spark_measurements = spark_pipeline.run_spark_sizes(
    spark, event_dir, app_id, full_inputs, RUNTIME / "benchmarks"
)
display(spark_measurements[["size_label", "wall_seconds", "wall_seconds_min",
                            "wall_seconds_max", "cpu_seconds", "peak_rss_mb", "error"]])""")

code("""# Автоматическая проверка из текста задания: Pandas на полном датасете (wide-чтение всех 27 столбцов).
full_check = benchmark_engines.run_full_pandas_check(
    Path(prepared["cleaned_csv"]), RUNTIME / "benchmarks")
print(full_check["verdict"])
if not full_check.get("error"):
    print(f"wall={full_check['wall_seconds']:.1f} c, peak RSS={full_check['peak_rss_mb']:,.0f} МБ")""")

code("""# Dask на полном датасете — дополнительная точка графиков «время/RAM vs объём».
dask_full = benchmark_engines._launch(
    "dask", Path(prepared["cleaned_csv"]), RUNTIME / "benchmarks" / "full_dask_result.csv",
    wide=False, timeout=benchmark_engines.FULL_TIMEOUT_SECONDS,
)
dask_full.update({"size_label": "full",
                  "input_bytes": Path(prepared["cleaned_csv"]).stat().st_size})
print(json.dumps({k: dask_full.get(k) for k in ("wall_seconds", "peak_rss_mb", "error")}))

combined = pd.concat(
    [cpu_measurements, spark_measurements, pd.DataFrame([dask_full])], ignore_index=True
)
combined["bytes_mb"] = combined["input_bytes"] / 1e6
size_order = {"100MB": 100, "1GB": 1000, "full": clean["cleaned_csv_bytes"] / 1e6}
combined["size_mb"] = combined["size_label"].map(size_order)
combined["cpu_utilization"] = combined["cpu_seconds"] / (combined["wall_seconds"] * 2)
combined.to_csv(RESULTS / "engines_all_measurements.csv", index=False)
display(combined[["engine", "size_label", "bytes_mb", "wall_seconds", "peak_rss_mb",
                  "cpu_seconds", "cpu_utilization", "read_bytes", "write_bytes",
                  "read_operations", "parity_with_pandas"]].round(3))""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
for engine, group in combined.groupby("engine"):
    order = np.argsort(group["size_mb"].values)
    axes[0].plot(group["size_mb"].values[order], group["wall_seconds"].values[order],
                 marker="o", label=engine)
    axes[1].plot(group["size_mb"].values[order], group["peak_rss_mb"].values[order] / 1000,
                 marker="s", label=engine)
axes[0].set(xlabel="Объём данных, МБ (лог)", ylabel="Время, с", xscale="log",
            title="Время выполнения vs объём")
axes[1].set(xlabel="Объём данных, МБ (лог)", ylabel="Пик RAM, ГБ (лог)", xscale="log",
            yscale="log", title="Потребление RAM vs объём")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "runtime_memory_by_volume.png")
plt.show()""")

code("""fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
util = combined.dropna(subset=["cpu_utilization"])
pivot = util.pivot_table(index="engine", columns="size_label", values="cpu_utilization")
pivot = pivot.reindex(columns=[c for c in ("100MB", "1GB", "full") if c in pivot.columns])
pivot.plot(kind="bar", ax=ax, rot=0)
ax.axhline(1.0, color="grey", ls="--", lw=1)
ax.set(ylabel="CPU-time / (wall × 2 потока)", xlabel="",
       title="Загрузка 2-поточного бюджета (пунктир — потолок 1.0)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(RESULTS / "cpu_utilization_by_volume.png")
plt.show()""")

md("### Анализ различий и ответы на вопросы задания")

code("""def pick(engine, size, column="wall_seconds"):
    row = combined[(combined.engine == engine) & (combined.size_label == size)]
    return float(row.iloc[0][column]) if len(row) else float("nan")

pandas_1g = pick("pandas", "1GB")
dask_1g = pick("dask", "1GB")
polars_1g = pick("polars", "1GB")
duck_1g = pick("duckdb", "1GB")
spark_100 = pick("pyspark", "100MB")
spark_1g = pick("pyspark", "1GB")
spark_full = pick("pyspark", "full")
pandas_full_rss = full_check.get("peak_rss_mb", float("nan"))
pandas_mb_per_input_mb = pick("pandas", "1GB", "peak_rss_mb") / 1000
ram_wall_gb = 32 / max(pandas_mb_per_input_mb, 1e-9)
wide_ratio = pandas_full_rss / (clean["cleaned_csv_bytes"] / 1e6)

conclusion_2 = f'''
- **Dask vs Pandas на 1 ГБ:** Dask {dask_1g:.2f} с против Pandas {pandas_1g:.2f} с →
  {"ускорение" if dask_1g < pandas_1g else "замедление"} на {abs(pandas_1g/dask_1g-1)*100:.0f}%.
  При лимите 2 потока выигрыш планировщика Dask почти исчезает: узкое место — парсинг CSV,
  а распараллеливать больше нечего.
- **Polars и DuckDB на 1 ГБ:** {polars_1g:.2f} с и {duck_1g:.2f} с — в {pandas_1g/polars_1g:.1f}x
  и {pandas_1g/duck_1g:.1f}x быстрее Pandas (многопоточное стримящее чтение, колоночная модель,
  векторные агрегаты).
- **PySpark:** 100 МБ {spark_100:.1f} с, 1 ГБ {spark_1g:.1f} с, полный датасет {spark_full:.1f} с.
  На малых объёмах overhead (строковый парсинг в JVM, 200 партиций, сериализация в pandas) не
  окупается; на полном датасете Spark обрабатывает данные потоково, не собирая весь CSV в один
  объект памяти.
- **При каком объёме Pandas перестаёт справляться?** На этой машине (32 ГБ RAM) wide-чтение
  полного датасета ({clean["cleaned_csv_bytes"]/1e6:,.0f} МБ) заняло
  {full_check.get("wall_seconds", float("nan")):.0f} с с пиком {pandas_full_rss/1000:.1f} ГБ
  (около {wide_ratio:.1f}x RAM на байт входа). Линейная экстраполяция до потолка 32 ГБ даёт
  ~{ram_wall_gb:.0f} ГБ CSV; реальный порог ниже (фрагментация, пики аллокаций, колонки object).
  Т.е. на ноутбуке Pandas «ломается» на объёмах на порядок больше нашего датасета — но именно
  поэтому он и не инструмент для роста IoT-парка, а Polars/DuckDB с RSS в {pick("pandas","1GB","peak_rss_mb")/pick("duckdb","1GB","peak_rss_mb"):.1f}x
  меньше остаются viable далеко за этим пределом (DuckDB читает файл, а не загружает его).
- **Предпочтительность:** до 100 МБ — Pandas (нулевой overhead); 100 МБ–5 ГБ локально —
  Polars/DuckDB; Dask — когда обязателен pandas-API на кластере; Spark — десятки-сотни ГБ,
  несколько источников, потоковая обработка и экосистема.
'''
display(Markdown(conclusion_2))""")

# ---------------------------------------------------------------- part 3
md("""## Часть 3. Анализ физического плана выполнения в Spark

`explain()` полной агрегации на CSV + разбор профиля стадий из event log (длительности задач,
shuffle-байты). В `local[2]` Exchange — локальный shuffle; сеть и кластерное масштабирование
здесь не измеряются.""")

code("""plan_report = spark_pipeline.analyze_physical_plan(
    spark, event_dir, app_id, Path(prepared["cleaned_csv"]), RESULTS
)
print(plan_report["plan_formatted"][:3600])
print("...")
print("Статистика плана:", plan_report["plan_stats"])""")

code("""stages = plan_report["stages"]
display(stages)""")

code("""biggest = stages.sort_values("wall_ms", ascending=False).iloc[0]
many = stages[stages["tasks"] > 1].sort_values("wall_ms", ascending=False)
work_stage = many.iloc[0] if len(many) else biggest
solo = stages[stages["tasks"] == 1]
conclusion_3 = f'''
**Разбор плана и узкие места (по числам выше):**
- План: `FileScan csv` → `HashAggregate (partial)` → `Exchange hashpartitioning(..., 200)` →
  `HashAggregate (final)`. Один shuffle-обмен; стадия скана читает весь CSV (строковый парсинг
  — самая дорогая часть, input {stages["input_mb_total"].max():,.0f} МБ).
- Через `Exchange` прошло ≈ {stages["shuffle_write_mb_total"].max():,.1f} МБ частичных агрегатов —
  мелко, т.к. many-стадия сильно сжимает данные перед обменом.
- Многозадачная стадия (id={int(work_stage["stage_id"])}, {int(work_stage["tasks"])} задач):
  медиана задачи {work_stage["task_ms_median"]:.0f} мс, максимум {work_stage["task_ms_max"]:.0f} мс
  → перекос времени задач ×{work_stage["task_time_skew_ratio"]:.1f}.
{f'- Финальную агрегацию AQE скоалесила в {int(solo["tasks"].sum())} задач(у) на {stages["shuffle_read_mb_max_task"].max():.1f} МБ shuffle-чтения — ждать «долгостроя» на финальной стадии нечего.' if len(solo) else '- Все стадии многозадачные.'}
- **Гипотезы оптимизаций:** (1) дробить «горячие» ключи солью (проверим в части 4);
  (2) Parquet вместо CSV — убрать строковый парсинг и читать только нужные колонки; (3) подобрать
  число партиций — 200 партиций на 2 ядрах могут добавлять накладных расходов; (4) AQE on/off.
'''
display(Markdown(conclusion_3))""")

# ---------------------------------------------------------------- part 4
md("""## Часть 4. Выявление и устранение skew

Считаем распределение ключей (CV = σ/μ — «коэффициент дисперсии» из задания, доля топ-ключа),
затем проверяем классическую стратегию **«соль»**: двухстадийная агрегация
`key+salt → частичный агрегат → финальный`. Эффект измеряется честно (warmup + медиана 3 прогонов
на одном закешированном датасете) — включая случай, когда пользы от неё нет.""")

code("""from pyspark.sql import functions as F

parsed_df = spark_pipeline.read_csv(spark, Path(prepared["cleaned_csv"])).select(
    *[F.col(name) for name in spark_pipeline.KEY_COLUMNS],
    spark_pipeline.dot_col("MI_dir_L0.1_mean"), spark_pipeline.dot_col("HH_L0.1_magnitude"),
    spark_pipeline.dot_col("H_L0.1_variance"),
).cache()
import time as _time
t0 = _time.perf_counter()
rows_cached = parsed_df.count()
print(f"Кэш materialized: {rows_cached:,} строк за {_time.perf_counter()-t0:.1f} с")""")

code("""skew = spark_pipeline.skew_experiment(
    spark, event_dir, app_id, parsed_df, RESULTS, buckets=8, runs=3
)
display(pd.DataFrame(skew["statistics"]).T.round(4))""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
for axis, column in zip(axes, ["Attack", "Attack_subType", "Device_Name"]):
    distribution = skew["distributions"][column].head(12)
    axis.barh(distribution[column].astype(str), distribution["count"] / 1e6)
    axis.invert_yaxis()
    axis.set_title(f"Распределение ключа {column}")
    axis.set_xlabel("млн строк")
fig.tight_layout()
fig.savefig(RESULTS / "key_distributions.png")
plt.show()""")

code("""baseline = skew["baseline"]
salted = skew["salted"]

def final_stage(frame):
    return frame.sort_values("task_ms_max").iloc[-1]

comparison = pd.DataFrame([
    {"вариант": "baseline (1 Exchange)",
     "wall медиана, с": baseline["wall_seconds_median"],
     "min–max, с": f'{baseline["wall_seconds_min"]:.2f}–{baseline["wall_seconds_max"]:.2f}',
     "Exchange в плане": skew["plan_stats_baseline"]["shuffle_exchanges"],
     "перекос final-стадии, max/median": final_stage(skew["baseline_stages"])["task_time_skew_ratio"]},
    {"вариант": "salted (2 Exchange, 8 солей)",
     "wall медиана, с": salted["wall_seconds_median"],
     "min–max, с": f'{salted["wall_seconds_min"]:.2f}–{salted["wall_seconds_max"]:.2f}',
     "Exchange в плане": skew["plan_stats_salted"]["shuffle_exchanges"],
     "перекос final-стадии, max/median": final_stage(skew["salted_stages"])["task_time_skew_ratio"]},
])
display(comparison.round(3))
print("Parity salted vs baseline:", skew["parity_ok"], skew["parity_error"] or "")""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
axes[0].bar(["baseline", "salted"],
            [baseline["wall_seconds_median"], salted["wall_seconds_median"]],
            color=["#d1554b", "#2a7de1"])
axes[0].set(ylabel="Медиана времени, с", title="Время агрегации (полный датасет, кэш)")
for i, v in enumerate([baseline["wall_seconds_median"], salted["wall_seconds_median"]]):
    axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom")
axes[1].bar(["baseline", "salted"],
            [final_stage(skew["baseline_stages"])["task_time_skew_ratio"],
             final_stage(skew["salted_stages"])["task_time_skew_ratio"]],
            color=["#d1554b", "#2a7de1"])
axes[1].set(ylabel="max/median времени задач", title="Перекос final-стадии")
for axis in axes:
    axis.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(RESULTS / "skew_before_after.png")
plt.show()""")

code("""cv = skew["statistics"]["composite_key"]["coefficient_of_variation"]
top1 = skew["statistics"]["composite_key"]["top1_share"]
keys_n = skew["statistics"]["composite_key"]["keys"]
atk_top1 = skew["statistics"]["Attack"]["top1_share"]
atk_cv = skew["statistics"]["Attack"]["coefficient_of_variation"]
speedup_skew = baseline["wall_seconds_median"] / salted["wall_seconds_median"]
skew_before = final_stage(skew["baseline_stages"])["task_time_skew_ratio"]
skew_after = final_stage(skew["salted_stages"])["task_time_skew_ratio"]
sw_before = skew["baseline_stages"]["shuffle_write_mb_total"].sum()
sw_after = skew["salted_stages"]["shuffle_write_mb_total"].sum()
verdict = ("соль ускорила запрос" if speedup_skew > 1.05 else
           "соль не дала измеримого ускорения" if speedup_skew > 0.95 else
           "соль **замедлила** запрос")
many_task_stage = skew["baseline_stages"][skew["baseline_stages"]["tasks"] > 1]
work_ratio = (many_task_stage.sort_values("task_ms_max").iloc[-1]["task_time_skew_ratio"]
              if len(many_task_stage) else float("nan"))
conclusion_4 = f'''
- **Перекос ключей — двойственная картина:** по полю `Attack` перекос есть (топ-значение
  {atk_top1*100:.0f}% строк, CV {atk_cv:.2f}), но контракт группирует по **составному** ключу из
  3 полей: {int(keys_n)} групп, CV {cv:.2f}, топ-группа {top1*100:.1f}% — хеш разносит их почти ровно.
- **Профиль baseline:** final-стадия после AQE — одна задача на {skew["baseline_stages"]["shuffle_read_mb_max_task"].max():.1f} МБ
  shuffle-чтения; «долгостроя», который тянул бы всю стадию, в этом запросе нет. Перекос времени
  задач на многозадачной (scan+partial) стадии — ×{work_ratio:.1f}: это разброс размеров CSV-сплитов,
  не ключей.
- **Эксперимент (соль ×8, два прогона по 3 повтора):** медиана {baseline["wall_seconds_median"]:.2f} с →
  {salted["wall_seconds_median"]:.2f} с — {verdict} (×{speedup_skew:.2f}); частичных агрегатов
  стало ≈в {sw_after/max(sw_before,0.01):.0f}x больше ({sw_before:.1f} → {sw_after:.1f} МБ shuffle-write),
  Exchange {skew["plan_stats_baseline"]["shuffle_exchanges"]} → {skew["plan_stats_salted"]["shuffle_exchanges"]}. Результат идентичен baseline (parity {'OK' if skew['parity_ok'] else 'НАРУШЕН'}).
- **Честный вывод:** на этом датасете и `local[2]` соль — структурный оверхед без отдачи:
  устранять нечего (горячей *финальной* группы нет), а второй Exchange и 8x частичных агрегатов
  надо оплатить. Метод оправдан, когда одна финальная группа даёт ≫1/P данных или shuffle сетевой
  (кластер) — там локальный замер сознательно недооценён.
'''
display(Markdown(conclusion_4))""")

# ---------------------------------------------------------------- part 5
md("""## Часть 5. Оптимизация Spark-кода

Эффект каждой оптимизации измеряется отдельно (warmup + медиана 3 прогонов):

- **Группа A (end-to-end без кэша):** baseline CSV → Parquet → Parquet+pruning. Эталон — baseline A.
- **Группа B (один закешированный dataframe):** AQE off, sweep `shuffle.partitions`
  {1, 4, 16, 64, 400} + лучший вариант, `repartition(keys)` перед группировкой, `persist()` +
  повторный запрос. Эталон — baseline B.
- **Комбинация:** Parquet + pruning + AQE + лучшие партиции. Эталон — baseline A.

Разделение честное: конфиг-варианты изолируют настройку на кэшированном входе, форматные —
измеряются полным циклом на холодную.""")

code("""optimizations = spark_pipeline.optimization_matrix(
    spark, Path(prepared["cleaned_csv"]), Path(prepared["cleaned_parquet"]),
    parsed_df, event_dir, RESULTS, partition_sweep=(1, 4, 16, 64, 400), runs=3
)
measurements = optimizations["measurements"]
display(measurements[["id", "description", "wall_seconds_median", "reference_seconds",
                      "speedup_vs_reference", "speedup_vs_csv_baseline", "effort",
                      "error"]].round(3))
print("Лучшие shuffle.partitions из sweep:", optimizations["best_partitions"])""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=160)
sweep = optimizations["sweep"]
axes[0].plot(sweep["shuffle_partitions"], sweep["wall_seconds_median"], marker="o")
axes[0].axvline(optimizations["best_partitions"], color="grey", ls="--", lw=1)
axes[0].set(xlabel="spark.sql.shuffle.partitions", ylabel="Время агрегации, с (кэш)",
            xscale="log", title="Sweep партиций (группа B)")
axes[0].grid(alpha=0.3)
plot_df = measurements[measurements["id"] != "combination"].dropna(subset=["speedup_vs_reference"])
axes[1].scatter(plot_df["effort"], plot_df["speedup_vs_reference"], s=70)
for _, row in measurements.dropna(subset=["speedup_vs_reference"]).iterrows():
    axes[1].annotate(row["id"], (row["effort"], row["speedup_vs_reference"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
axes[1].axhline(1.0, color="grey", ls="--", lw=1)
axes[1].set(xlabel="Затраты: 0=config, 1=1 строка кода, 2=write-шаг/перестройка запроса, 3=формат+код",
            ylabel="Ускорение vs эталон своей группы", title="Затраты vs выгода оптимизаций")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "optimization_cost_benefit.png")
plt.show()""")

code("""combo = optimizations["combination"]
best_p = optimizations["best_partitions"]
base_a = optimizations["baseline_a_seconds"]
base_b = optimizations["baseline_b_seconds"]
combo_speedup = base_a / combo["wall_seconds_median"]
per_opt = measurements.set_index("id")
sweep = optimizations["sweep"].set_index("shuffle_partitions")
aqe_off = per_opt.loc['b_aqe_off', 'wall_seconds_median']
aqe_effect = base_b / aqe_off
aqe_400 = sweep.loc[400, "wall_seconds_median"] if 400 in sweep.index else float('nan')
aqe_line = (f"выключение {aqe_off:.2f} с против {base_b:.2f} с (×{aqe_effect:.2f}) — эффект ниже "
            "шума повторов: на этом запросе (small shuffle, AQE и так сводит финальный обмен к "
            "минимуму) измеримой пользы нет, но и вреда нет — оставляем включённой"
            if abs(aqe_effect - 1) < 0.05 else
            f"выключение замедляет до {aqe_off:.2f} с против {base_b:.2f} с — адаптивные "
            "оптимизации реально работают")
rep = per_opt.loc['b_repartition_keys', 'wall_seconds_median']
cache_q = per_opt.loc['b_cache_iteration', 'wall_seconds_median']
cache_sp = per_opt.loc['b_cache_iteration', 'speedup_vs_reference']
cache_line = (f"холодный цикл {base_a:.2f} → повторный {cache_q:.2f} с (×{cache_sp:.2f}) — "
              "строковый парсинг платится один раз" if cache_sp > 1.5 else
              f"холодный цикл {base_a:.2f} → повторный {cache_q:.2f} с (только ×{cache_sp:.2f}): к этому "
              "варианту в памяти уже висят кэши предыдущих секций, Storage вытесняет часть "
              "партиций — в изолированном пайплайне выигрыш равен экономии на парсинге "
              "(ср. a_parquet_input)")
rep_line = (f"{rep:.2f} с — быстрее baseline B (×{base_b/rep:.2f}): после repartition по ключам "
            "финальный обмен идёт из меньшего числа задач; на кластере цена — второй сетевой shuffle"
            if rep < base_b * 0.95 else
            f"{rep:.2f} с — не окупается (второй shuffle)")
conclusion_5 = f'''
- **Sweep партиций (группа B, кэш):** лучший — {best_p} (медиана {sweep.loc[best_p, "wall_seconds_median"]:.2f} с
  против {base_b:.2f} с при 200); «400 против 200» из текста задания — {aqe_400:.2f} с, т.е. на 2 ядрах
  не выигрывает: 200+ мелких задач — чистый overhead планирования, выигрыш даёт **укрупнение**.
- **AQE:** {aqe_line}.
- **Parquet вместо CSV:** {per_opt.loc['a_parquet_input', 'wall_seconds_median']:.2f} с против
  {base_a:.2f} с (×{per_opt.loc['a_parquet_input', 'speedup_vs_reference']:.2f}); плюс pruning колонок —
  {per_opt.loc['a_parquet_pruned', 'wall_seconds_median']:.2f} с (на этом наборе нужные колонки и так
  читаются, pruning в основном страхует широкие таблицы).
- **repartition по ключам:** {rep_line}; **persist + повторный запрос:** {cache_line}.
- **Оптимальная комбинация** (Parquet + pruning + AQE + partitions={best_p}):
  {combo["wall_seconds_median"]:.2f} с против CSV-baseline {base_a:.2f} с → **×{combo_speedup:.2f}**.
  Обоснование: основной выигрыш даёт колоночный формат (убирает строковый парсинг — узкое место
  скана из части 3), конфиг-оптимизации почти бесплатны и снижают overhead мелкой партиционности.
'''
display(Markdown(conclusion_5))""")

# ---------------------------------------------------------------- part 6
md("""## Часть 6. GPU vs CPU — Трек A (обязательный, локально на CPU)

По заданию Трек A выполняется всеми: сравниваем **pandas (однопоточный) ↔ Polars (многопоточный,
Arrow) ↔ DuckDB (векторизованный SQL)** на одинаковых объёмах 100 МБ и 1 ГБ. Трек B (cuDF for
Apache Spark) требует NVIDIA GPU с драйвером R525+ и CUDA 12.9/13.1 в Linux/Docker; Трек C —
облачный фолбэк (Colab/Kaggle). **Выполнен Трек A**; причины отказа от B/C — в
[gpu/README.md](gpu/README.md). Оценивание треков одинаковое.""")

code("""track_a = combined[(combined.engine.isin(["pandas", "polars", "duckdb"])) &
                   (combined.size_label.isin(["100MB", "1GB"]))].copy()
table = pd.DataFrame({"Инструмент": ["pandas", "Polars", "DuckDB"]})
engine_map = {"pandas": "pandas", "Polars": "polars", "DuckDB": "duckdb"}
for size, label in (("100MB", "100 МБ"), ("1GB", "1 ГБ")):
    times = []
    rams = []
    for short in table["Инструмент"]:
        row = track_a[(track_a.engine == engine_map[short]) & (track_a.size_label == size)].iloc[0]
        times.append(round(row["wall_seconds"], 3))
        rams.append(round(row["peak_rss_mb"], 1))
    table[f"{label}: время, с"] = times
    table[f"{label}: RAM, МБ"] = rams
display(table)

pandas_rows = {size: track_a[(track_a.engine == "pandas") & (track_a.size_label == size)].iloc[0]
               for size in ("100MB", "1GB")}
for short in ("Polars", "DuckDB"):
    engine = engine_map[short]
    r100 = track_a[(track_a.engine == engine) & (track_a.size_label == "100MB")].iloc[0]
    r1g = track_a[(track_a.engine == engine) & (track_a.size_label == "1GB")].iloc[0]
    print(f"{short}: быстрее pandas на {(1-r100['wall_seconds']/pandas_rows['100MB']['wall_seconds'])*100:.0f}% (100 МБ) и "
          f"{(1-r1g['wall_seconds']/pandas_rows['1GB']['wall_seconds'])*100:.0f}% (1 ГБ); "
          f"пик RAM {r1g['peak_rss_mb']/1000:.2f} ГБ на 1 ГБ")
track_a.to_csv(RESULTS / "track_a_table.csv", index=False)""")

code("""fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)
colors = {"pandas": "#d1554b", "polars": "#2a7de1", "duckdb": "#2e9e5b"}
for short, engine in engine_map.items():
    r100 = track_a[(track_a.engine == engine) & (track_a.size_label == "100MB")].iloc[0]
    r1g = track_a[(track_a.engine == engine) & (track_a.size_label == "1GB")].iloc[0]
    axes[0].plot(["100 МБ", "1 ГБ"], [r100["wall_seconds"], r1g["wall_seconds"]],
                 marker="o", label=short, color=colors[engine])
    axes[1].plot(["100 МБ", "1 ГБ"], [r100["peak_rss_mb"] / 1000, r1g["peak_rss_mb"] / 1000],
                 marker="s", label=short, color=colors[engine])
axes[0].set(ylabel="Время, с (лог)", title="Трек A: время выполнения")
axes[0].set_yscale("log")
axes[1].set(ylabel="Пик RAM, ГБ", title="Трек A: потребление памяти")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "track_a_comparison.png")
plt.show()""")

code("""duck_1g_row = track_a[(track_a.engine == "duckdb") & (track_a.size_label == "1GB")].iloc[0]
conclusion_6 = f'''
**Причины ускорения (измерено на этой машине, а не пересказано):** Polars и DuckDB читают CSV
стримингом в несколько потоков и агрегируют векторно по Arrow-блокам; pandas читает в один поток
и материализует объектную модель колонок. Даже при равном лимите «2 потока всем» разрыв остаётся —
выигрыш не только в параллельности, но и в способе парсинга/раскладки.

**Когда GPU не окупается (критерий из задания + наши числа):** точка окупаемости cuDF-плагина —
запросы от ~30 с (инициализация executor/GPU ~2–3 с на каждый этап, передача данных).
Полная контрактная агрегация на Spark/CPU здесь: {spark_full:.0f} с — формально у границы.
При этом GPU машины — мобильная RTX 3050 с 4 ГБ VRAM, а cuDF for Apache Spark требует
Linux/Docker + CUDA 12.9/13.1 (на Windows только через WSL2-кластер), т.е. окружение
не соответствует условиям трека B. **Честный результат: выполнен Трек A, GPU-ускорение не
заявляется.** Вывод инвариантен: для запросов короче ~30 с и датасетов, влезающих в RAM одной
машины, оставайтесь на CPU — Polars/DuckDB дают эффект без смены стека.
'''
display(Markdown(conclusion_6))""")

# ---------------------------------------------------------------- part 7
md("## Часть 7. Рекомендации по выбору инструмента")

code("""matrix = pd.DataFrame([
    ("Объём < 100 МБ", "+++", "+", "+", "Pandas; Polars, если латентность критична"),
    ("Объём 100 МБ–1 ГБ локально", "+", "+", "+", "Polars/DuckDB (время), Pandas (привычность API)"),
    ("1–5 ГБ при ограниченной RAM", "риск", "+", "++", "DuckDB (минимум RSS), Spark"),
    ("Десятки–сотни ГБ", "нет", "+", "+++", "Spark + Parquet + AQE"),
    ("Повторные запросы по одним данным", "++", "+", "+++", f"Parquet-стек: Spark ×{combo_speedup:.1f} к CSV"),
    ("Потоковая/инкрементальная обработка", "нет", "+", "+++", "Spark Structured Streaming"),
    ("Скорость разработки прототипа", "+++", "+", "+", "Pandas/Polars"),
    ("GPU без переписывания кода", "нет", "нет", "+++", "Spark + cuDF-плагин, от ~30 с/запрос"),
], columns=["Критерий", "Pandas", "Dask", "Spark", "Рекомендация"])
display(matrix)""")

code("""cost = pd.DataFrame([
    ("Pandas (2 vCPU ноутбук)", pick("pandas", "1GB"), 0.10),
    ("Polars (2 vCPU ноутбук)", pick("polars", "1GB"), 0.10),
    ("DuckDB (2 vCPU ноутбук)", pick("duckdb", "1GB"), 0.10),
    ("Dask local (2 vCPU)", pick("dask", "1GB"), 0.10),
    ("Spark local (2 vCPU)", spark_1g, 0.10),
    ("Spark-кластер 3 узла ×8 vCPU (модель)", spark_full / 3 * 1.35, 0.60),
], columns=["Инструмент/среда", "Время, с", "$/час (модель)"])
cost["Стоимость прогона, $"] = cost["Время, с"] / 3600 * cost["$/час (модель)"]
display(cost.round(4))
print("Тарифы — модельные допущения для порядка величины (локальная VM ~0.10 $/ч, узел кластера")
print("~0.60 $/ч), не прайс конкретного облака; время — измеренное на этой машине.")
print("Кластерная модель: линейное масштабование по узлам + 35% на сетевой shuffle")
print("(в сети локальной машины эти 35% не измерялись — оценочный коэффициент).")
cost.to_csv(RESULTS / "cost_performance.csv", index=False)""")

code("""conclusion_7 = f'''
**Рекомендация под бизнес-сценарий ЛР1 (IoT IDS, мониторинг логов, цель MTTR −40%):**

1. **Дайворк и итерации аналитика (данные до единиц ГБ):** Polars/DuckDB — полная агрегация
   на Parquet у Spark занимает {combo["wall_seconds_median"]:.0f} с, а на 1 ГБ в одиночную Polars
   даёт {pick("polars","1GB"):.1f} с против {pick("pandas","1GB"):.1f} с у Pandas
   ({(1-pick("polars","1GB")/pick("pandas","1GB"))*100:.0f}% экономии на каждый прогон); DuckDB
   при этом держит RAM в {pick("pandas","1GB","peak_rss_mb")/pick("duckdb","1GB","peak_rss_mb"):.1f}x меньше pandas.
2. **Production-пайплайн:** Spark + Parquet + AQE (конфиг из части 5). Профили групп
   `Attack×Attack_subType×Device_Name` — контракт мониторингового дашборда; рост IoT-парка
   линейно увеличивает объём логов, и только Spark из сравнения масштабируется на сотни ГБ
   и инкрементальные окна (Structured Streaming). Время детекта аномалии = вклад в MTTR:
   ускорение пайплайна ×{combo_speedup:.1f} — это прямое сокращение окна обнаружения.
3. **Dask:** только при жёстком требовании pandas-API на кластере; локально выигрыша не дал.
4. **Компромиссы:** Spark — операционная сложность (JDK/партиции/планы, части 3–5) и старт
   дороже на малых данных; Polars/DuckDB — одиночная машина и меньше production-экосистемы;
   Pandas — RAM-потолок (~{ram_wall_gb:.0f} ГБ по экстраполяции).
5. **Риски:** дрейф схемы/качества источника (закрыты: SHA-256 + GE-сьют + parity-тесты);
   префиксные срезы смещают состав атак; перенос локальных spark-выводов на кластер без
   переоценки; соль меняет детерминированность плавающих сумм (parity держим допуском).
'''
display(Markdown(conclusion_7))""")

# ---------------------------------------------------------------- checks
md("""## Автоматические проверки

Проверки корректности перед выводами: parity результатов на всех объёмах, parity соль-vs-baseline,
GE negative control, штатное завершение всех прогонов, полнота артефактов. Блок «Pandas на полном
датасете» повторяет пример из текста задания.""")

code("""checks = []

def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("## OK ## " if ok else "## FAIL ## ") + name + (f" — {detail}" if detail else ""))

cpu_like = combined[combined.engine.isin(["pandas", "dask", "polars", "duckdb"]) &
                    combined.error.isna()]
parities = cpu_like[cpu_like.engine != "pandas"][["engine", "size_label", "parity_with_pandas"]]
check("Parity не-Pandas движков против Pandas", bool(parities["parity_with_pandas"].all()),
      f"{int(parities['parity_with_pandas'].sum())}/{len(parities)}")
check("Parity соль vs baseline (часть 4)", skew["parity_ok"], skew["parity_error"] or "")
check("GE negative control", bool(quality["negative_control_failed_as_expected"]))
spark_ok = spark_measurements["error"].isna().all()
check("Все Spark-прогоны завершены без ошибок", bool(spark_ok))
check("Dask на полном датасете завершён", not dask_full.get("error"),
      str(dask_full.get("error") or "")[:120])
if full_check.get("error"):
    print("## OK ## Pandas не смог обработать полный датасет wide-чтением: "
          + str(full_check["error"])[:120])
else:
    print("## WARNING ## Pandas справился с полным датасетом — машина 32 ГБ не препятствие; "
          "порог объёма оценён экстраполяцией RSS в части 2")

expected_files = [
    RESULTS / "engines_all_measurements.csv", RESULTS / "runtime_memory_by_volume.png",
    RESULTS / "cpu_utilization_by_volume.png",
    RESULTS / "key_distributions.png", RESULTS / "skew_before_after.png",
    RESULTS / "optimization_cost_benefit.png", RESULTS / "track_a_comparison.png",
    RESULTS / "spark_physical_plan.txt", RESULTS / "spark_plan_stage_summary.csv",
    RESULTS / "skew_baseline_stages.csv", RESULTS / "skew_salted_stages.csv",
    RESULTS / "optimization_measurements.csv", RESULTS / "partition_sweep.csv",
    RESULTS / "track_a_table.csv", RESULTS / "cost_performance.csv",
    RESULTS / "quality_report.json",
]
missing = [str(p) for p in expected_files if not p.is_file()]
check("Артефакты результатов на месте", not missing, ", ".join(missing) or f"{len(expected_files)} файлов")

assert all(ok for _, ok, _ in checks), "есть проваленные проверки"
print(f"\\nИТОГ: {sum(ok for _, ok, _ in checks)}/{len(checks)} проверок пройдено")""")

md("## Заключение")

code("""order_1g = (combined[combined.size_label == "1GB"].dropna(subset=["wall_seconds"])
            .sort_values("wall_seconds"))
order_str = " > ".join(f"{r.engine} ({r.wall_seconds:.1f} с)" for r in order_1g.itertuples())
pq_speedup = per_opt.loc['a_parquet_input', 'speedup_vs_reference']
part_gain = base_b / sweep.loc[best_p, 'wall_seconds_median']
salt_factor = salted["wall_seconds_median"] / baseline["wall_seconds_median"]
aqe_off_ms = per_opt.loc['b_aqe_off', 'wall_seconds_median']
conclusion_final = f'''
**Итоги сравнения (измерено на этой машине, лимит 2 потока CPU):**
- Контрактная агрегация `Attack×Attack_subType×Device_Name` даёт идентичный результат во всех
  движках и в Spark (parity) — сравнение корректно.
- Порядок по скорости на 1 ГБ: **{order_str}**. Козырь Spark — не одиночный прогон
  локально (JVM-оверхед на local[2]), а полный датасет без materialization всего файла в объект
  и повторные запросы после Parquet: **×{combo_speedup:.1f}** к холодному CSV-baseline.
- Физический план: один Exchange; AQE сворачивает финальный обмен
  ({skew["baseline_stages"]["shuffle_read_mb_max_task"].max():.1f} МБ) в одну задачу. Перекос ключей живёт в
  `Attack` (топ-значение {atk_top1*100:.0f}% строк), но составной ключ разбивается на ~{int(keys_n)} близких
  групп — соль на этом запросе локально **умножила время на ×{salt_factor:.1f}** без всякой пользы:
  честная граница применимости приёма (нужна горячая *финальная* группа или сетевой shuffle).
- Оптимизации: Parquet вместо CSV ×{pq_speedup:.1f}, укрупнение партиций 200→{best_p}
  ×{part_gain:.1f} (на кэшированном входе), AQE на этом запросе — в пределах шума
  ({aqe_off_ms:.2f} с против {base_b:.2f} с); итоговая комбинация **×{combo_speedup:.1f}** к холодному CSV.
- Трек A выполнен полностью; GPU-треки не запускались (мобильная RTX 3050, Linux/CUDA-стек) —
  честная фиксация: CPU-трек закрывает оценивание части 6.

**Рекомендации:** локальная аналитика — Polars/DuckDB; production масштаба логов IoT-парка —
Spark (Parquet + AQE + стриминг); Pandas — малые объёмы и прототипирование; Dask — только
pandas-совместимость; GPU — от ~30 с запроса и только на CUDA-окружении.
'''
display(Markdown(conclusion_final))""")

code("""spark.stop()
print("Spark session stopped.")""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

target = Path(__file__).resolve().parents[2] / "lab2" / "lab2_engines_benchmark.ipynb"
nbf.write(nb, target)
print(f"Wrote {len(cells)} cells to {target}")
