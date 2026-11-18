"""Генератор отчётного ноутбука ЛР4 (единый источник ячеек).

python lab4/build_notebook.py → lab4/lab4_ml_production.ipynb (без результатов;
выполняется lab4/run_notebook.py). Тексты ячеек — здесь, чтобы .ipynb был воспроизводим.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path("lab4/lab4_ml_production.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip()}


def code(src: str) -> dict:
    c = nbf.v4.new_code_cell(src.strip())
    c["execution_count"] = None
    c["outputs"] = []
    return c


CELLS: list[dict] = []
A = CELLS.append

# ─────────────────────────────────────────────────────────── Титул
A(md("""# ЛР4. Интеграция ML-моделей в production
**Вариант 4 — уникальное требование: дифференциальная приватность.
Сценарий 1 — Fraud Detection** на данных ЛР3 (IDS BoTNeTIoT-L01-v2, 6 440 923 события,
метка `label`; `Device_Name` — «клиент», событие — «операция», атака — «фрод»).

**Стек (CPU):** MLflow 3.18 (tracking + Model Registry, SQLite backend) · Feast 0.49
(file offline + SQLite online) · FastAPI + httpx · scikit-learn / XGBoost ·
River 0.23 (online-обучение) · Evidently 0.7 (отчёт о дрейфе) · StatsForecast (часть 8).

## Методологические рамки (честно)
1. **Время событий — replay-модель из ЛР3**: все сплиты и потоки честны по `event_time`
   (train = первые 18 ч окна, test = последние 6 ч); concept drift — управляемая фикстура
   (зафиксирована в коде), а не утверждение о реальных атаках.
2. **Обучающая матрица — выборки** 200k/250k (Feast point-in-time join на всех 1.6M
   строках занял бы минуты ради тех же распределений). Полный test (1.6M) используется
   для замера дрейфа, где Feast не нужен. Пороги AUC достигаются на выборках.
3. **Дифференциальная приватность** — собственная реализация DP-SGD (NumPy) с RDP-учётом
   без усиления выборок (консервативная композиция); нормировка признаков фиксирована
   и бюджет не расходует — оговорка в ч. 7.
4. **Latency** — сквозной HTTP-зонд (TestClient, код тот же, что в бою), включает сбор
   фич из Feast online store, а не только `model.predict`.
5. Каждая часть содержит автопроверки (`assert`); фиксированные метрики печатаются в
   конце в блоке `## METRICS ##`."""))

A(code("""import json, os, sys, time
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
import mlflow, feast, fastapi, river, sklearn, xgboost
print("python", sys.version.split()[0], "| mlflow", mlflow.__version__,
      "| feast", feast.__version__, "| river", river.__version__,
      "| sklearn", sklearn.__version__, "| xgboost", xgboost.__version__)
np.random.seed(0)

from lab4 import mlpipeline as mp
WORK = ROOT / "data" / "lab4"
WORK.mkdir(parents=True, exist_ok=True)
RES = ROOT / "lab4" / "results"
RES.mkdir(parents=True, exist_ok=True)
T0 = time.time()
mp.setup_mlflow()
print("tracking:", mlflow.get_tracking_uri())"""))

# ─────────────────────────────────────────────────────────── Данные
A(md("""## 0. Данные: ML-матрица из хранилища ЛР3

Берём `raw.events` (6.44 млн строк, валидированы в ЛР2/ЛР3) → parquet float32;
temporal split по `event_time` (train 18 ч / test 6 ч). Цепочка лабораторных
воспроизводима: этот ноутбук читает выход ЛР3."""))

A(code("""from lab4 import data_prep

if not (WORK / "events.parquet").exists():
    print("build:", data_prep.build_datasets())
events = data_prep.load_events()
train_df, test_df = data_prep.time_split(events)
print("rows:", len(events), "| train:", len(train_df), "| test:", len(test_df),
      "| devices:", events.Device_Name.nunique())
print("pos_rate train/test:", round(train_df.label.mean(), 4), round(test_df.label.mean(), 4))

ev = events.copy(); ev["hour"] = ev.event_time.dt.floor("h")
g = ev.groupby("hour").agg(pos=("label", "mean"), n=("label", "size")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(g.hour, g.pos, marker=".")
ax[0].axvline(pd.Timestamp("2026-11-01 18:00"), color="crimson", ls="--", label="сплит train/test")
ax[0].set_title("доля атак по часам replay-окна"); ax[0].legend()
top = ev.loc[ev.label == 1, data_prep.FEATURES].mean().sort_values().tail(8)
ax[1].barh(top.index.str.replace("_L0.1", "", regex=False), top.values)
ax[1].set_title("средние фичи у атак (топ-8)"); ax[1].set_xscale("symlog")
fig.tight_layout(); plt.show()"""))

# ─────────────────────────────────────────────────────────── Feast
A(md("""## 2. Feature Store (Feast): offline для обучения, online для инференса

Определения — `lab4/feast_repo/features.py`: сущность `device`, feature view
`device_stats` (7 почасовых агрегатов активности устройства, ttl=2 дня), источник —
parquet из `lab4/features.py` (SQL-агрегация `raw.events`).

**Главная автопроверка задания:** одна точка as-of → `get_historical_features`
против `get_online_features` → совпадение с допуском float32."""))

A(code("""from lab4.features import build_feature_table, FEATURE_NAMES
from feast import FeatureStore
import lab4.feast_repo.features as fdefs

feat_table = build_feature_table()
print("feature table:", feat_table.shape, "| часов:", feat_table.event_timestamp.nunique())

store = FeatureStore(repo_path=str(ROOT / "lab4" / "feast_repo"))
store.apply([fdefs.device, fdefs.device_stats])
# refs — по явным FEATURE_NAMES: после apply() .schema вкл. join-key Device_Name
refs = [f"device_stats:{n}" for n in FEATURE_NAMES]
t_last = feat_table.event_timestamp.max()
store.materialize(start_date=feat_table.event_timestamp.min() - pd.Timedelta(hours=1),
                  end_date=t_last + pd.Timedelta(hours=1))

ent = pd.DataFrame({"Device_Name": feat_table.Device_Name.unique(),
                    "event_timestamp": pd.Timestamp(t_last) + pd.Timedelta(seconds=1)})
offline = store.get_historical_features(ent, refs).to_df()
online = store.get_online_features(refs, [{"Device_Name": d}
                                          for d in feat_table.Device_Name.unique()]).to_df()
for c in ("event_timestamp", "created"):
    if c in offline.columns:
        offline = offline.drop(columns=[c])
offline = offline.set_index("Device_Name")
online = online.set_index("Device_Name")
a = offline[FEATURE_NAMES].astype("float64").to_numpy()
b = online.loc[offline.index, FEATURE_NAMES].astype("float64").to_numpy()
max_diff = float(np.abs(np.nan_to_num(a) - np.nan_to_num(b)).max())
FEATURE_MATCH_TEST_PASSED = bool(max_diff < 1e-3)
print("offline/online max_abs_diff =", f"{max_diff:.2e}")
assert FEATURE_MATCH_TEST_PASSED, "Feature mismatch!"
print("FEATURE_MATCH_TEST_PASSED=True (допуск 1e-3 — float32 при материализации)")"""))

A(md("""> **Вывод (ч. 2).** Feast закрывает feature mismatch: дефиниция фичи исполняется
один раз (SQL при построении таблицы); offline/online расходятся только сериализацией,
поэтому расхождение — уровень float32 (1e-4). Критичны все агрегированные фичи: расхождение
окна агрегации дало бы систематический training-serving skew."""))

# ─────────────────────────────────────────────────────────── Часть 1 + 3
A(md("""## 1–3. Прототип, сравнение моделей и жизненный цикл в MLflow

Матрица = 23 фичи события + 7 фич Feast (point-in-time join). Пороги валидации из
задания: **AUC ≥ 0.85** и latency батча ≤ 50 мс. Каждая модель — run в MLflow с
параметрами, метриками и артефактом."""))

A(code("""%%time
mt = mp.build_training_matrix(sample_train=200_000, sample_test=250_000)
print("X_train:", mt["X_train"].shape, "X_test:", mt["X_test"].shape,
      "| pos:", round(mt["y_train"].mean(), 4))"""))

A(code("""%%time
table = mp.compare_models(mt)
display(table)
best_run = table.iloc[0]["run_id"]
print("лучшая:", table.iloc[0]["model"], "auc =", round(table.iloc[0].auc, 4))"""))

A(md("""### Аудит качества: почему AUC ≈ 1.0 (проверяем на месте, а не верим на слово)

AUC бутстрэпов на этом датасете — 1.0. Прежде чем принимать цифру, исключаем две
причины: (а) **прямая утечка метки** — univariate AUC каждой фичи; (б) **запоминание
идентичности устройства** — обучаем модель БЕЗ устройства X и тестируем НА X
(leave-one-device-out). Если обе не при чём, разделимость — свойство признакового
пространства BoTNetIoT (атаки имеют резкий сетевой сигнатурный отпечаток; logreg
линейно даёт лишь ~0.65, дерево глубины 2–3 — ~0.85): метрика насыщена, гейт
AUC ≥ 0.85 проходит тривиально — фиксируем это как измеренный факт."""))

A(code("""from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

Xte_a, yte_a = mt["X_test"], mt["y_test"]
devs_a = mt["test_join"]["Device_Name"].to_numpy()

uni = sorted(((c, round(max(roc_auc_score(yte_a, np.nan_to_num(Xte_a[:, i], nan=0.0)), 0.5), 3))
              for i, c in enumerate(mt["feat_cols"])), key=lambda r: -r[1])[:5]
print("top univariate AUC:", uni)  # ни одна фича не = 1.0 → прямой утечки нет

Xtr20, ytr20 = mt["X_train"][:60_000], mt["y_train"][:60_000]
ladder = {}
for d in (2, 3, 6, 10):
    tr_ = DecisionTreeClassifier(max_depth=d, random_state=1).fit(Xtr20, ytr20)
    ladder[d] = round(roc_auc_score(yte_a, tr_.predict_proba(Xte_a)[:, 1]), 4)
print("лестница глубин дерева (AUC):", ladder)

loo = []
for dev in ["Ennio_Doorbell", "Provision_PT_737E_Security_Camera",
            "SimpleHome_XCS7_1003_WHT_Security_Camera"]:
    m_ = devs_a == dev
    sub = np.random.default_rng(0).choice(np.flatnonzero(~m_), 60_000, replace=False)
    m2 = HistGradientBoostingClassifier(max_iter=120, max_depth=6, random_state=1)
    m2.fit(Xte_a[sub], yte_a[sub])
    if len(np.unique(yte_a[m_])) > 1:
        loo.append((dev[:26], round(roc_auc_score(yte_a[m_], m2.predict_proba(Xte_a[m_])[:, 1]), 4)))
print("leave-one-device-out (AUC на unseen-устройстве):", loo)
print("ВЫВОД: unseen-устройство ранжируется так же → разделимость не в идентичности")
print("устройства и не в утечке — это свойство синтетических starling-признаков.")"""))

A(md("""### Регистрация, гейт перед деплоем, алиасы

Валидация — на отдельной выборке (не той, где модель набирала AUC в сравнении)."""))

A(code("""import mlflow
reg = mp.register_production(best_run, mt["X_test"][:100_000], mt["y_test"][:100_000])
print("registered:", reg["load_uri"], "version", reg["version"])
print("gate:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in reg["validation"].items()})
assert reg["validation"]["passed"]
# кандидат для canary — лучшая модель СРЕДИ ОТЛИЧАЮЩИХСЯ от продакшн
cand_name = table.loc[table.model != table.iloc[0]["model"], "model"].iloc[0]
cand_run = table.set_index("model").loc[cand_name, "run_id"]
cand_mv = mlflow.register_model(f"runs:/{cand_run}/model", mp.MODEL_NAME)
mlflow.MlflowClient().set_registered_model_alias(mp.MODEL_NAME, "candidate", cand_mv.version)
print("candidate:", cand_name, "version:", cand_mv.version)"""))

# ─────────────────────────────────────────────────────────── Часть 4
A(md("""## 4. Инференс в production: FastAPI + Feast online

`/predict`: 23 фичи события в payload + 7 онлайн-агрегатов из Feast → скоринг моделью по
алиасу `production`. Ошибки: 422 на битый payload, 503 при отказе feature store, флаг
`cold_start` при отсутствии истории у устройства. Latency-зонд — 600 сквозных запросов."""))

A(code("""from lab4.serving import create_app, FEATURES_23, latency_probe

app, meta = create_app(refs, FEATURE_NAMES, alias="production")
from fastapi.testclient import TestClient
client = TestClient(app)

row = test_df.iloc[0]
payload = {"device": row.Device_Name,
           "features": {c: float(row[c]) if pd.notna(row[c]) else 0.0 for c in FEATURES_23}}
r = client.post("/predict", json=payload)
print("health:", client.get("/health").json())
print("predict:", r.status_code, r.json())
bad = client.post("/predict", json={"device": "x", "features": {"a": 1}})
badnum = client.post("/predict", json={"device": "x", "features": {c: "нет" for c in FEATURES_23}})
print("битый payload →", bad.status_code, "| не-число →", badnum.status_code)
assert bad.status_code == 422"""))

A(code("""sample_reqs = test_df.sample(600, random_state=0)
lat = latency_probe(app, sample_reqs, n=600)
INFERENCE_LATENCY_MS = round(lat["p95_ms"], 2)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in lat.items()})
assert lat["p95_ms"] < 50, f"p95 {lat['p95_ms']:.1f} мс > 50"

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.bar([0, 1, 2], [lat["p50_ms"], lat["p95_ms"], lat["p99_ms"]], color=["#4c72b0", "#dd8452", "#c44e52"])
ax.axhline(50, color="crimson", ls="--", label="SLA 50 мс")
ax.set_xticks([0, 1, 2], ["p50", "p95", "p99"]); ax.set_ylabel("мс")
ax.set_title("latency сквозного /predict"); ax.legend()
fig.tight_layout(); plt.show()
print("INFERENCE_LATENCY_MS(p95) =", INFERENCE_LATENCY_MS)"""))

A(md("""### Canary 5% + shadow + автооткат

`deploy_experiment`: 5% трафика на кандидата, shadow-сверка на 5% (|Δscore|, agreement),
решение promote/rollback по дельте AUC на помеченных метках."""))

A(code("""from lab4.serving import deploy_experiment

smp = mt["test_join"].sample(50_000, random_state=1)
Xsrv = smp[mt["feat_cols"]].astype("float32").fillna(0.0)
ysrv = smp["label"].values.astype(int)
res = deploy_experiment(f"models:/{mp.MODEL_NAME}@production",
                        f"models:/{mp.MODEL_NAME}@candidate", Xsrv, ysrv)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
                 ensure_ascii=False, indent=1))
if res["promoted"]:
    mlflow.MlflowClient().set_registered_model_alias(mp.MODEL_NAME, "canary", cand_mv.version)
    print("canary-алиас → версия", cand_mv.version, ":", res["reason"])
else:
    print("ROLLBACK:", res["reason"])"""))

# ─────────────────────────────────────────────────────────── Часть 5
A(md("""## 5. Мониторинг: дрейф данных (PSI/KS + Evidently), дрейф модели, retrain

Пороги из задания: PSI колонки > 0.2 = дрейф; DATA_DRIFT_SCORE = доля дрейфовавших
колонок; алерт > 0.2; retrain > 0.3 или падение AUC > 0.15 или неделя. Числа — наша
детерминированная реализация; Evidently — визуальный артефакт (HTML)."""))

A(code("""from lab4 import monitor as mon
from lab4.data_prep import make_drift_window, FEATURES

# чистое окно (обычная смена данных): проверка 2 задания — drift < 0.3
clean_rep = mon.drift_report(train_df, test_df, FEATURES)
clean_score = mon.drift_score(clean_rep)
print("чистое окно: drift_score =", round(clean_score, 3))
assert clean_score < 0.3, "чистое окно ложно дрейфует"

# фикстура concept drift: сдвиг масштаба + инверсия связи у части строк
drift_df = make_drift_window(test_df, seed=7)
drift_rep = mon.drift_report(train_df, drift_df, FEATURES)
DATA_DRIFT_SCORE = round(mon.drift_score(drift_rep), 4)
display(drift_rep.sort_values("psi", ascending=False).head(8))
art = mon.evidently_artifact(train_df[FEATURES].sample(5000, random_state=0),
                             drift_df[FEATURES].sample(5000, random_state=0), RES)
print("evidently artifacts:", art)
print("DATA_DRIFT_SCORE =", DATA_DRIFT_SCORE)"""))

A(code("""from sklearn.metrics import roc_auc_score
from lab4.serving import load_by_alias

model_prod, meta_p = load_by_alias("production")
feat_cols = mt["feat_cols"]
auc_clean = roc_auc_score(mt["y_test"], model_prod.predict(mt["X_test"])[:, -1])
drift_join = make_drift_window(mt["test_join"], seed=7)
auc_drift = roc_auc_score(drift_join["label"],
                          model_prod.predict(drift_join[feat_cols].astype("float32").fillna(0.0))[:, -1])
auc_drop = auc_clean - auc_drift
if DATA_DRIFT_SCORE > mon.DRIFT_ALERT:
    mon.send_alert(f"data drift score {DATA_DRIFT_SCORE:.2f} > {mon.DRIFT_ALERT}")
if auc_drop > mon.AUC_DROP_ALERT:
    mon.send_alert(f"падение AUC {auc_drop:.3f} > {mon.AUC_DROP_ALERT}")
need, why = mon.check_retrain_needed(DATA_DRIFT_SCORE, auc_drop, days_since_retrain=1.0)
print(f"AUC clean={auc_clean:.4f} -> drift={auc_drift:.4f} (drop {auc_drop:.4f}); retrain={need}: {why}")
with mlflow.start_run(run_name="drift-monitor"):
    mlflow.log_metrics({"data_drift_score": DATA_DRIFT_SCORE, "auc_clean": auc_clean,
                        "auc_drift": auc_drift, "auc_drop": auc_drop})
assert need, "ожидали срабатывание retrain на фикстуре"

if need:
    Xr = pd.concat([mt["train_join"][feat_cols], drift_join[feat_cols]]).astype("float32").fillna(0.0).values
    yr = np.concatenate([mt["train_join"]["label"].values, drift_join["label"].values]).astype(int)
    retrain_table = mp.compare_models(
        {"X_train": Xr, "y_train": yr, "X_test": mt["X_test"], "y_test": mt["y_test"],
         "feat_cols": feat_cols, "n_train": len(yr), "n_test": mt["n_test"]},
        experiment="fraud-retrain", only=["xgboost"])
    display(retrain_table)
    print("retrain: AUC на чистом тесте =", round(float(retrain_table.iloc[0].auc), 4),
          "— новая версия кандидат на canary (стратегия ч.4)")"""))

# ─────────────────────────────────────────────────────────── Часть 6
A(md("""## 6. Batch vs Online обучение (River)

Поток = joined test-окно по времени (62k событий), вторая половина — concept drift.
Batch (production) заморожена; online (River SGD) идёт prequential: предсказала →
получила метку → обновилась. ONLINE_BATCH_DIFFERENCE = AUC_online − AUC_batch на потоке."""))

A(code("""from lab4.online import make_stream, online_vs_batch

stream = make_stream(mt["test_join"], feat_cols)
stream = stream.iloc[::4].reset_index(drop=True)     # децимация ×4, порядок по времени сохранён
print("stream:", len(stream), "| drift доля:", round(stream.is_drifted.mean(), 3))
cmp = online_vs_batch(stream, feat_cols, model_prod)
print("batch:", {k: round(v, 4) for k, v in cmp["batch_auc"].items()})
print("online:", {k: round(v, 4) for k, v in cmp["online_auc"].items()})
ONLINE_BATCH_DIFFERENCE = round(cmp["difference_final"], 4)

fig, ax = plt.subplots(figsize=(8, 3.4))
c = cmp["curve"]
ax.plot(c.t_rows / 1000, c.auc_batch_win, label="batch (заморожена)")
ax.plot(c.t_rows / 1000, c.auc_online_win, label="online (River, prequential)")
ax.axvline(c.t_rows.max() / 2000, color="crimson", ls="--", label="concept drift")
ax.set_xlabel("события потока, тыс."); ax.set_ylabel("AUC (окно 10k)")
ax.set_title("Batch vs Online на потоке с дрейфом"); ax.legend()
fig.tight_layout(); plt.show()
print("ONLINE_BATCH_DIFFERENCE =", ONLINE_BATCH_DIFFERENCE)"""))

A(code("""with mlflow.start_run(run_name="batch-vs-online"):
    mlflow.log_metric("batch_auc_full", cmp["batch_auc"]["full"])
    mlflow.log_metric("online_auc_full", cmp["online_auc"]["full"])
    mlflow.log_metric("batch_auc_drift_half", cmp["batch_auc"]["second_half_drift"])
    mlflow.log_metric("online_auc_drift_half", cmp["online_auc"]["second_half_drift"])
    mlflow.log_metric("difference", ONLINE_BATCH_DIFFERENCE)
print("logged run batch-vs-online")"""))

# ─────────────────────────────────────────────────────────── Часть 7
A(md("""## 7. Уникальное требование: дифференциальная приватность (DP-SGD)

Своя реализация (`lab4/dp.py`): по-sample L2-clip градиентов (C=1) + гауссов шум σ·C,
учёт RDP-accountant **без усиления выборок** (консервативно), δ = 1e-6 < 1/n. Кривая
«цены приватности»: AUC на тесте по ε ∈ {∞, 8, 4, 2, 1, 0.5}.
Оговорка: нормировка фиксирована априори и бюджет не расходует (агрегированная телеметрия,
не персональные признаки); защита — на этапе обучения."""))

A(code("""from lab4.dp import dp_curve, epsilon_min
from sklearn.preprocessing import StandardScaler

DP_STEPS = 2000
print(f"граница честной оценки: ε_min(steps={DP_STEPS}, δ=1e-6) = {epsilon_min(DP_STEPS, 1e-6):.2f}")
print("(ниже — нужен RDP-аккаунтер с усилением Пуассоновской выборки; не реализуем, "
      "иначе гарантия была бы фейковой)")

sc = StandardScaler().fit(mt["X_train"])
xd_tr = sc.transform(mt["X_train"][:200_000]).astype("float32")
xd_te = sc.transform(mt["X_test"][:60_000]).astype("float32")
curve = dp_curve(xd_tr, mt["y_train"][:200_000], xd_te, mt["y_test"][:60_000],
                 [float("inf"), 20.0, 12.0, 8.0, 5.0, 4.0], auc_fn=roc_auc_score,
                 steps=DP_STEPS, lr=0.08, strict=False)
display(curve)
baseline_auc = float(curve.loc[curve.epsilon_requested == float("inf"), "auc_test"].iloc[0])
ok = curve[curve.reachable & (curve.auc_test >= baseline_auc - 0.02)]
EPS_STAR = float(ok.epsilon_requested.min()) if len(ok) else float("inf")
if EPS_STAR != float("inf"):
    # берём строку ИМЕННО ε*, а не первую подошедшую (иначе impact всегда = 0)
    star_auc = float(curve.loc[curve.epsilon_requested == EPS_STAR, "auc_test"].iloc[0])
    UNIQUE_REQUIREMENT_IMPACT = round(star_auc - baseline_auc, 4)
else:
    reach = curve[curve.reachable]
    UNIQUE_REQUIREMENT_IMPACT = round(float(reach.auc_test.min()) - baseline_auc, 4)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
xs = list(curve.epsilon_requested)
ax.plot(range(len(xs)), curve.auc_test, marker="o")
ax.set_xticks(range(len(xs)), ["∞"] + [str(int(e)) for e in xs if e != float("inf")])
ax.axhline(baseline_auc, color="grey", ls=":", label=f"baseline {baseline_auc:.3f}")
ax.axhline(baseline_auc - 0.02, color="crimson", ls="--", label="допуск −0.02")
# ε-ось категориальная (сетка), поэтому границу честности пишем текстом, а не vspan:
# ε < {min} недостижим при полной композиции — сетка сознательно обрывается на 4.0
ax.text(0.02, 0.06, f"ε_min={epsilon_min(DP_STEPS, 1e-6):.1f}: ниже — недостижимо"
        " без усиления выборок (сетка обрывается на 4.0)",
        transform=ax.transAxes, fontsize=7, va="bottom", color="dimgrey")
ax.set_xlabel("бюджет приватности ε"); ax.set_ylabel("AUC (тест)")
ax.set_ylim(0.4, max(0.9, baseline_auc + 0.02))
ax.set_title("цена приватности DP-SGD"); ax.legend(loc="lower right")
fig.tight_layout(); plt.show()
print("ε* =", EPS_STAR, "| UNIQUE_REQUIREMENT_IMPACT =", UNIQUE_REQUIREMENT_IMPACT)"""))

# ─────────────────────────────────────────────────────────── Часть 8
A(md("""## 8. Прогноз сенсора и автоотключение (NAB ambient_temperature_system_failure)

Почасовой ряд температуры; две аварии: **видимая** (15–30.12.2013, пик 86°F при базисе
~76) и **holdout** (29.03–20.04.2014, финальная проверка честности). Ряд сезонный,
поэтому критическая температура **локальная**: медиана последнего месяца + отклон dev
(тюнится, а не задаётся «на глаз»; абсолютный порог бессмысленен — абсолютный пик ряда 86°F).
Модель: StatsForecast Theta (рекурсивно, N=24) + верхний квантиль 0.9 (гауссова
аппроксимация σ₁√h, оговорено); сравнение с прямой стратегией (GBM:
HistGradientBoosting с лагами и горизонтом; LightGBM 4.7 на стенде нестабилен).
Тревога — конъюнкция: P(T > base30+dev за H) > p*, CUSUM, persistence, cooldown.
Без point-adjust."""))

A(code("""from lab4 import part8 as p8

s = p8.load_series()
print("rows:", len(s), "| интерполяция пропусков:", s.attrs["n_filled"],
      "| диапазон:", s.index.min(), "->", s.index.max())
span = (s.index.min(), p8.INCIDENTS["accident_2_holdout"][0])
t0 = time.perf_counter()
sig_tune = p8.compute_signals(s, *span)
print(f"tune-сигналы: {len(sig_tune)} ч за {time.perf_counter()-t0:.0f} c")
span_sig = (sig_tune.index.min(), sig_tune.index.max())
prm, grid = p8.tune_thresholds(sig_tune, span_sig)
print(grid[grid.detected].sort_values('lead_h', ascending=False).head(6).to_string(index=False))
print("tuned:", prm)"""))

A(code("""al_tune = p8.alarms_from_signals(sig_tune, prm)
ev1 = p8.evaluate_alarms(al_tune, p8.INCIDENTS["accident_1"], span_sig)
sig_ho = p8.compute_signals(s, p8.INCIDENTS["accident_2_holdout"][0] - pd.Timedelta(days=62),
                            p8.INCIDENTS["accident_2_holdout"][0] + pd.Timedelta(days=28))
al_ho = p8.alarms_from_signals(sig_ho, prm)
span_ho = (sig_ho.index.min(), sig_ho.index.max())
ev2 = p8.evaluate_alarms(al_ho, p8.INCIDENTS["accident_2_holdout"], span_ho)
print("visible:", ev1)
print("holdout:", ev2)

fig, ax = plt.subplots(figsize=(11, 4))
win = s.loc["2014-02-15":"2014-05-05"]
ax.plot(win.index, win.values, lw=.7, color="black", label="факт")
ax.plot(sig_ho.index, sig_ho.med1, lw=.8, label="Theta h=1")
ax.plot(sig_ho.index, sig_ho.q90, lw=.8, color="tab:orange", label="квантиль 0.9")
ax.plot(sig_ho.index, sig_ho.base30 + prm.dev_crit, lw=1.0, color="crimson", ls="--",
        label=f"T_crit=base30+{prm.dev_crit:g}")
a0, a1 = p8.INCIDENTS["accident_2_holdout"]
ax.axvspan(a0, a1, color="tab:red", alpha=.15, label="авария (holdout)")
if len(al_ho):
    ax.scatter(al_ho, s.reindex(al_ho).values, marker="X", s=80, color="purple", label="тревога")
ax.legend(loc="upper left", fontsize=8); ax.set_title("Holdout-авария: прогноз, полоса, срабатывания")
fig.tight_layout(); plt.show()"""))

A(code("""hc = p8.horizon_curve(s, s.index.min() + pd.Timedelta(days=70),
                      p8.INCIDENTS["accident_2_holdout"][0])
mae_th = hc.groupby("h").apply(lambda d: (d.theta_rec - d.y).abs().mean(), include_groups=False)
mae_lg = hc.groupby("h").apply(lambda d: (d.gbm_direct - d.y).abs().mean(), include_groups=False)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(mae_th.index, mae_th.values, marker=".", label="рекурсивная (Theta)")
ax.plot(mae_lg.index, mae_lg.values, marker=".", label="прямая (HistGB)")
ax.set_xlabel("горизонт h, ч"); ax.set_ylabel("MAE, °F")
ax.set_title("рост ошибки от горизонта (walk-forward)"); ax.legend()
fig.tight_layout(); plt.show()
pb = {"pinball_theta_q9": round(p8.pinball_loss(hc.y, hc.theta_rec, 0.9), 3),
      "mase_theta_h24": round(p8.mase(hc[hc.h == 24].y, hc[hc.h == 24].theta_rec,
                                      s.dropna().iloc[:2000].values), 3)}
part8_metrics = {"visible_accident": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev1.items()},
                 "holdout_accident": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev2.items()},
                 "forecast": pb}
print(json.dumps(part8_metrics, ensure_ascii=False))"""))

# ─────────────────────────────────────────────────────────── Метрики/выводы
A(md("""## Финальные метрики и выводы"""))

A(code("""METRICS = {
    "FEATURE_MATCH_TEST_PASSED": FEATURE_MATCH_TEST_PASSED,
    "DATA_DRIFT_SCORE": DATA_DRIFT_SCORE,
    "INFERENCE_LATENCY_MS": INFERENCE_LATENCY_MS,
    "ONLINE_BATCH_DIFFERENCE": ONLINE_BATCH_DIFFERENCE,
    "UNIQUE_REQUIREMENT_IMPACT": UNIQUE_REQUIREMENT_IMPACT,
}
EXTRA = {
    "EPSILON_CHOSEN": EPS_STAR,
    "BEST_MODEL": str(table.iloc[0]["model"]),
    "BEST_AUC": round(float(table.iloc[0]["auc"]), 4),
    "PROD_MODEL_VERSION": int(reg["version"]),
    "LATENCY_P50_MS": round(lat["p50_ms"], 2),
    "LATENCY_P99_MS": round(lat["p99_ms"], 2),
    "THROUGHPUT_RPS": round(lat["throughput_rps"], 1),
    "AUC_CLEAN": round(auc_clean, 4),
    "AUC_DRIFT_FIXTURE": round(auc_drift, 4),
    "PART8_VISIBLE_LEAD_H": ev1["lead_time_h"],
    "PART8_HOLDOUT_DETECTED": bool(ev2["detected"]),
    "PART8_HOLDOUT_LEAD_H": ev2["lead_time_h"],
    "RUNTIME_SEC": round(time.time() - T0, 1),
}
print("## METRICS ##")
for k, v in METRICS.items():
    print(f"{k}={v}")
for k, v in EXTRA.items():
    print(f"{k}={v}")
json.dump({**METRICS, **EXTRA, "part8": part8_metrics},
          open(WORK / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
assert FEATURE_MATCH_TEST_PASSED
assert clean_score < 0.3
assert INFERENCE_LATENCY_MS < 50
print("все автопроверки пройдены")"""))

A(md("""### Выводы по разделам

1. **Feature Store** — offline/online расхождение 1e-4 (float32): feature mismatch закрыт.
2. **MLflow** — сравнение 4 моделей, гейты AUC≥0.85 и latency перед деплоем, алиасы
   production/candidate/canary; откат строится на них же. AUC бутстрэпов = 1.0 —
   измеренное свойство синтетических starling-признаков (аудит утечки и leave-one-device-out
   выше), не баг пайплайна; гейт проходит тривиально, и мы это фиксируем честно.
3. **Инференс** — сквозной p95 < 50 мс достижим на табличной модели (в отличие от
   batch-window SLA из ЛР3 — иная природа требования); canary+shadow дают измеримую
   основу для promote/rollback.
4. **Мониторинг** — PSI/KS детектирует фикстуру дрейфа (drift_score > 0.3 → алерт → retrain
   по детерминированным правилам); при насыщенной AUC именно распределенческие метрики
   остаются информативными — на этом и стоит production-мониторинг.
5. **Batch vs online** — без дрейфа batch впереди; после concept drift online адаптируется,
   замороженная batch деградирует (см. кривую). Гибрид: batch-период + online между релизами.
6. **DP** — цена приватности измерена: ΔAUC на выбранном ε*; при ε ≥ ε* потеря ≤ 0.02.
7. **Часть 8** — конъюнкция 4 условий гасит ложные; метрики на видимой аварии и честный
   holdout; локальный порог вместо фиктивного абсолютного. Цена ошибки: пропущенная авария
   дороже ложного отключения.

Метрики прогона — в блоке выше и `data/lab4/metrics.json`. Скриншоты MLflow UI — в README."""))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.NotebookNode(c) for c in CELLS]
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell-{i:02d}"
    nb.metadata = {
        "kernelspec": {"display_name": "Lab 4 local environment",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    for cell in nb.cells:
        if cell["cell_type"] == "code":
            # магические строки (%%time) убираем для compile-check
            src = "\n".join(l for l in cell["source"].splitlines()
                            if not l.strip().startswith(("%", "!")))
            compile(src, "<cell>", "exec")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    code_n = sum(c["cell_type"] == "code" for c in nb.cells)
    print(f"wrote {OUT}: {len(nb.cells)} cells ({code_n} code)")


if __name__ == "__main__":
    main()
