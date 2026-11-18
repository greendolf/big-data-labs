"""ЛР4: инференс в production — FastAPI + Feast online + безопасность деплоя.

Стек запроса /predict: событие (23 фичи) + онлайн-агрегаты устройства из Feast
(7 фичей) → модель по alias из Model Registry. Всё измеримо: latency-зонд,
canary 5%, shadow-режим, автооткат по метрике.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Модель запроса на УРОВНЕ модуля: при from __future__ import annotations
    FastAPI резолвит строковые аннотации только по globals модуля — локальный
    класс он не находит и трактует тело запроса как query-параметр (422)."""
    device: str = Field(..., min_length=1, description="Device_Name")
    features: dict[str, float] = Field(..., description="23 фичи события")

FEATURES_23 = [
    "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std", "HH_L0.1_magnitude",
    "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
    "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std", "HpHp_L0.1_magnitude",
    "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc",
]
MODEL_NAME = "FraudModel"


def load_by_alias(alias: str = "production") -> tuple[Any, dict[str, Any]]:
    """pyfunc-модель по алиасу + её версия из реестра."""
    import mlflow

    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    return model, {"alias": alias, "version": mv.version, "run_id": mv.run_id}


class FeatureAssembler:
    """Собирает вектор модели: 23 фичи события + 7 онлайн-агрегатов Feast.

    Холодный старт: нет данных в online-сторе → NULLы замаплены на 0 (агрегаты
    «нулевой активности»), флаг cold_start=True в ответе — честная деградация.
    """

    def __init__(self, feast_refs: list[str], agg_names: list[str]) -> None:
        from feast import FeatureStore
        from pathlib import Path

        self.store = FeatureStore(repo_path=str(Path.cwd() / "lab4" / "feast_repo"))
        self.refs = feast_refs
        self.agg_names = agg_names

    def online(self, device: str) -> tuple[np.ndarray, bool]:
        row = self.store.get_online_features(self.refs, [{"Device_Name": device}]).to_dict()
        vec = np.array([row.get(n, [None])[0] for n in self.agg_names], dtype="float32")
        cold = bool(np.isnan(vec).any())
        return np.nan_to_num(vec, nan=0.0), cold

    def matrix(self, events: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """events: колонки FEATURES_23 + Device_Name → DataFrame с 30 колонками модели."""
        aggs, cold_flags = [], []
        for dev in events["Device_Name"].unique():
            v, cold = self.online(dev)
            aggs.append((dev, dict(zip(self.agg_names, v))))
            cold_flags.append(cold)
        amap = dict(aggs)
        out = events[FEATURES_23].astype("float32").copy()
        for name in self.agg_names:
            out[name] = events["Device_Name"].map(lambda d, n=name: amap[d][n]).astype("float32")
        return out, np.array(cold_flags)


def create_app(feast_refs: list[str], agg_names: list[str], alias: str = "production"):
    """FastAPI-приложение. Инференс-путь и обработка ошибок по требованиям задания."""
    from fastapi import FastAPI, HTTPException

    model, meta = load_by_alias(alias)
    asm = FeatureAssembler(feast_refs, agg_names)

    app = FastAPI(title="Lab4 fraud scoring", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": MODEL_NAME, **meta}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        missing = [c for c in FEATURES_23 if c not in req.features]
        if missing:
            raise HTTPException(status_code=422,
                                detail=f"не хватает {len(missing)} фич: {missing[:3]}...")
        row = pd.DataFrame([{**{c: req.features[c] for c in FEATURES_23},
                             "Device_Name": req.device}])
        t0 = time.perf_counter()
        try:
            X, cold = asm.matrix(row)
        except Exception as exc:  # Feast/IO деградация — не 500 без объяснения
            raise HTTPException(status_code=503, detail=f"feature store: {exc}") from exc
        score = float(model.predict(X)[:, -1][0])
        return {"score": score, "prediction": int(score >= 0.5),
                "model_version": meta["version"], "cold_start": bool(cold[0]),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    return app, meta


def latency_probe(app, requests_df: pd.DataFrame, n: int = 600, seed: int = 0) -> dict[str, Any]:
    """Нагрузочный зонд через TestClient (in-process, один писатель Feast)."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(requests_df), size=n, replace=False)
    latencies, errors = [], 0
    t_start = time.perf_counter()
    for i in idx:
        row = requests_df.iloc[int(i)]
        feats = row[FEATURES_23].astype("float64").fillna(0.0)  # NaN в JSON нельзя
        payload = {"device": row["Device_Name"],
                   "features": {c: float(feats[c]) for c in FEATURES_23}}
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        if r.status_code != 200:
            errors += 1
    wall = time.perf_counter() - t_start
    a = np.array(latencies)
    return {"n": n, "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()), "throughput_rps": n / wall,
            "error_rate": errors / n}


def deploy_experiment(prod_uri: str, cand_uri: str, X_eval: pd.DataFrame,
                      y_eval: np.ndarray, canary_share: float = 0.05,
                      seed: int = 11, min_auc_delta: float = -0.02) -> dict[str, Any]:
    """Canary 5% + shadow-сверка + автооткат. Возвращает измеренный исход.

    shadow: обе модели на подвыборке → mean |Δscore| и agreement@0.5;
    canary: 5% трафика на кандидат, метрика — AUC кандидата против контролера
    на помеченных данных; кандидат побеждает, если ΔAUC >= min_auc_delta.
    """
    import mlflow
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    prod = mlflow.pyfunc.load_model(prod_uri)
    cand = mlflow.pyfunc.load_model(cand_uri)
    n = len(X_eval)
    route_c = rng.random(n) < canary_share

    scores_p = prod.predict(X_eval)[:, -1]
    scores_c = cand.predict(X_eval)[:, -1]
    shadow_mask = rng.random(n) < 0.05
    delta = np.abs(scores_p - scores_c)[shadow_mask]
    agree = float(((scores_p > 0.5) == (scores_c > 0.5)).mean())

    auc_p = roc_auc_score(y_eval, scores_p)
    auc_c = roc_auc_score(y_eval, scores_c)
    promoted = (auc_c - auc_p) >= min_auc_delta
    return {"canary_share": canary_share, "canary_n": int(route_c.sum()),
            "auc_prod": auc_p, "auc_canary": auc_c,
            "shadow_mean_abs_delta": float(delta.mean()), "shadow_agreement": agree,
            "promoted": bool(promoted),
            "rollback": (not promoted),
            "reason": ("AUC кандидата не хуже контролера" if promoted else
                       f"AUC кандидата ниже на {auc_p - auc_c:.4f} — откат")}
