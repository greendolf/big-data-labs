"""ЛР4: обучение моделей, MLflow tracking и Model Registry.

Один источник правды для ноутбука, smoke-теста и CI:
  build_training_matrix() — честный temporal split + Feast offline-фичи;
  compare_models()        — 4 модели в MLflow, сводная таблица;
  register_production()   — регистрация лучшей + валидация (AUC/latency) + alias production.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path.cwd()
WORK = ROOT / "data" / "lab4"
TRACKING_DB = WORK / "mlflow.db"
ARTIFACTS = WORK / "artifacts"

MODEL_NAME = "FraudModel"


def setup_mlflow() -> None:
    import mlflow

    WORK.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # MLflow 3: при sqlite tracking URI реестр наследует его автоматически.
    # set_registry_uri("database-optimized") в 3.18 резолвится в file-store
    # (maintenance-mode исключение) — не зовём.
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")


def use_experiment(name: str):
    """Эксперимент с ЯВНЫМ artifact_location (иначе sqlite-бэкленд пишет mlruns в cwd)."""
    import mlflow

    try:
        mlflow.get_experiment_by_name(name)
    except Exception:
        pass
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(name)
    if existing is None:
        client.create_experiment(name=name, artifact_location=ARTIFACTS.as_uri())
    return mlflow.set_experiment(name)


def build_training_matrix(sample_train: int = 200_000, sample_test: int = 250_000,
                          seed: int = 1) -> dict[str, Any]:
    """train/test по времени + почасовые Feast-агрегаты (point-in-time, без leakage).

    Feast file-source point-in-time join линеен по строкам (~5к строк/с на этой машине),
    поэтому обе выборки ограничены: честный AUC на 200k/250k, а не на всех 1.6М (иначе
    join занимает минуты ради тех же самых распределений). Полный test остаётся доступен
    через load_events/time_split для монитора дрейфа (там Feast не нужен).
    Возвращаем и numpy-матрицы (сравнение моделей) и joined-фреймы (serving/online).
    """
    from lab4.data_prep import load_events, time_split, FEATURES
    from lab4.features import FEATURE_NAMES

    df = load_events()
    tr, te = time_split(df)
    tr_s = tr.sample(min(sample_train, len(tr)), random_state=seed)
    te_s = te.sample(min(sample_test, len(te)), random_state=seed)

    from feast import FeatureStore

    store = FeatureStore(repo_path=str(ROOT / "lab4" / "feast_repo"))
    refs = [f"device_stats:{n}" for n in FEATURE_NAMES]

    def with_fs(part: pd.DataFrame) -> pd.DataFrame:
        p = (part.reset_index(drop=True)
             .rename(columns={"event_time": "event_timestamp"}))
        ent = p[["Device_Name", "event_timestamp"]].drop_duplicates()
        fs = store.get_historical_features(ent, refs).to_df()
        # Feast отдаёт ts как UTC-aware; источник — naive wall-clock → снимаем tz
        if str(fs["event_timestamp"].dtype).endswith(", UTC]"):
            fs["event_timestamp"] = fs["event_timestamp"].dt.tz_localize(None)
        # right уникален по ключам → левый merge сохраняет ровно строки part
        return p.merge(fs, on=["Device_Name", "event_timestamp"], how="left")

    tr_fs, te_fs = with_fs(tr_s), with_fs(te_s)
    feat_cols = FEATURES + FEATURE_NAMES
    Xtr = tr_fs[feat_cols].astype("float32").fillna(0.0).values
    Xte = te_fs[feat_cols].astype("float32").fillna(0.0).values
    return {
        "X_train": Xtr, "y_train": tr_fs["label"].values.astype(int),
        "X_test": Xte, "y_test": te_fs["label"].values.astype(int),
        "train_join": tr_fs, "test_join": te_fs,
        "feat_cols": feat_cols, "n_train": len(tr_fs), "n_test": len(te_fs),
    }


def _models() -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "logreg": LogisticRegression(max_iter=500, n_jobs=4),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=24, n_jobs=-1, random_state=1),
        "xgboost": XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=8,
                                 eval_metric="auc", tree_method="hist"),
        "hist_gb": HistGradientBoostingClassifier(max_iter=250, max_depth=8, random_state=1),
    }


def compare_models(mt: dict[str, Any], experiment: str = "fraud-model-selection",
                   only: list[str] | None = None) -> pd.DataFrame:
    import mlflow
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    use_experiment(experiment)
    models = _models()
    if only:
        models = {k: v for k, v in models.items() if k in only}
    rows = []
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            t0 = time.perf_counter()
            model.fit(mt["X_train"], mt["y_train"])
            train_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            proba = model.predict_proba(mt["X_test"])[:, 1]
            infer_ms = (time.perf_counter() - t1) * 1000 / len(mt["X_test"])
            auc = roc_auc_score(mt["y_test"], proba)
            ap = average_precision_score(mt["y_test"], proba)
            f1 = f1_score(mt["y_test"], (proba > 0.5))
            mlflow.log_params({"model": name, "n_train": mt["n_train"]})
            mlflow.log_metrics({"auc": auc, "avg_precision": ap, "f1": f1,
                                "train_sec": train_s, "infer_ms_per_row": infer_ms})
            # логируем обёртку: pyfunc.predict → матрица вероятностей (n, k),
            # а не метки класса — иначе весь serving/monitor/AUC-контур сломается
            from lab4.wrap import ProbaWrapper

            mlflow.sklearn.log_model(ProbaWrapper(model), name="model",
                                     serialization_format="cloudpickle")
            rows.append({"model": name, "auc": auc, "avg_precision": ap, "f1": f1,
                         "train_sec": round(train_s, 1),
                         "infer_ms_per_row": round(infer_ms, 5),
                         "run_id": mlflow.active_run().info.run_id})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def validate_for_deploy(model, X_test, y_test, *, min_auc=0.85, max_batch_ms=50.0) -> dict[str, Any]:
    """Валидация перед деплоем (по заданию): AUC-порог и latency-порог.

    model — pyfunc, залогированный через ProbaWrapper: predict → (n, k) вероятности.
    """
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_test, model.predict(X_test)[:, 1])
    # латентность на батче из 1000 (инференс-партия), не на одной строке
    idx = np.random.default_rng(0).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    t0 = time.perf_counter()
    model.predict(X_test[idx])
    batch_ms = (time.perf_counter() - t0) * 1000
    ok = auc >= min_auc and batch_ms <= max_batch_ms
    return {"auc": auc, "batch1000_ms": batch_ms, "passed": ok,
            "min_auc": min_auc, "max_batch_ms": max_batch_ms}


def register_production(run_id: str, X_val, y_val) -> dict[str, Any]:
    """Регистрация модели рана в Model Registry + валидация + alias production.

    Валидация — на ОТДЕЛЬНОЙ выборке (не той, где модель набирала AUC в сравнении).
    """
    import mlflow

    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    val = validate_for_deploy(model, X_val, y_val)
    if not val["passed"]:
        raise ValueError(f"Валидация не пройдена: {val}")
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
    return {"version": mv.version, "validation": val, "alias": "production",
            "load_uri": f"models:/{MODEL_NAME}@production"}
