"""CI-smoke для lab4: самодостаточный end-to-end на синтетике (боевых данных не требует).

Проверяет связку компонентов, а не качество на 6.4M строк:
  фичи-таблица → Feast apply → historical/online allclose → обучение logreg →
  MLflow sqlite + регистрация + alias → FastAPI /predict + latency → PSI-дрейф →
  DP-обучение на игрушечных данных → часть 8 чистые функции.
Время: ~1–2 мин после установки стека.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path.cwd()

from pydantic import BaseModel


class _SmokeReq(BaseModel):
    """Pydantic-модель на УРОВНЕ модуля: FastAPI не видит BaseModel, объявленный
    внутри функции, и трактует его как query-параметр → 422 на валидном JSON."""
    x: list[float]


def main() -> int:
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="lab4_smoke_"))
    repo = ROOT / "lab4" / "feast_repo"
    ok = True
    try:
        import numpy as np
        import pandas as pd

        # --- 1) фичи-таблица для Feast (мини-аналог build_feature_table) ------
        rng = np.random.default_rng(0)
        hours = pd.date_range("2026-11-01", periods=6, freq="h")
        rows = [{"Device_Name": d, "event_timestamp": h,
                 "event_count_1h": rng.random() * 100,
                 "avg_magnitude_1h": rng.random() * 10, "avg_radius_1h": rng.random() * 10,
                 "avg_mi_mean_1h": rng.random() * 10, "std_mi_mean_1h": rng.random(),
                 "avg_weight_1h": rng.random() * 10, "p95_mag_1h": rng.random() * 12,
                 "created": pd.Timestamp("2026-11-01 06:00")}
                for d in ("dev-a", "dev-b") for h in hours]
        ft = pd.DataFrame(rows)
        ft.to_parquet(repo / "device_hourly_stats.parquet", index=False)

        from feast import FeatureStore
        sys.path.insert(0, str(repo))
        import features as fdefs  # noqa: F401

        store = FeatureStore(repo_path=str(repo))
        store.apply([fdefs.device, fdefs.device_stats])
        # ВАЖНО: после apply() .schema у FeatureView включает join-key Device_Name —
        # refs строим по явным именам агрегатов, не по .schema
        from lab4.features import FEATURE_NAMES
        refs = [f"device_stats:{n}" for n in FEATURE_NAMES]
        t_last = ft.event_timestamp.max()
        store.materialize(start_date=ft.event_timestamp.min() - pd.Timedelta(hours=1),
                          end_date=t_last + pd.Timedelta(hours=1))
        ent = pd.DataFrame({"Device_Name": ["dev-a", "dev-b"],
                            "event_timestamp": t_last + pd.Timedelta(seconds=1)})
        off = store.get_historical_features(ent, refs).to_df().set_index("Device_Name")
        on = store.get_online_features(refs, [{"Device_Name": "dev-a"},
                                              {"Device_Name": "dev-b"}]).to_df()
        agg_cols = [c for c in off.columns if c not in ("Device_Name", "event_timestamp")]
        assert off["avg_magnitude_1h"].notna().all(), "historical пустой"
        d_max = float(np.abs(off.loc[["dev-a", "dev-b"], agg_cols].to_numpy(dtype="float64")
                             - on.set_index("Device_Name").loc[["dev-a", "dev-b"], agg_cols]
                             .to_numpy(dtype="float64")).max())
        assert d_max < 1e-4, f"feature mismatch {d_max}"
        print(f"[1/6] FEAST ok: offline/online diff={d_max:.2e}")

        # --- 2) обучение logreg + MLflow + регистрация ------------------------
        import mlflow
        import mlflow.sklearn
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        db = tmp / "mf.db"
        # MLflow 3: registry унаследует sqlite tracking URI; alias "database-optimized"
        # в 3.18 резолвится в file-store и ловит maintenance-mode — не используем
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        client = mlflow.MlflowClient()
        art = tmp / "artifacts"
        exp = client.create_experiment(name="smoke", artifact_location=art.as_uri())
        mlflow.set_experiment("smoke")

        X = rng.normal(size=(400, 6))
        y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.4, size=400) > 0).astype(int)
        model = LogisticRegression(max_iter=300).fit(X, y)
        # логируем ProbaWrapper (как в бою): pyfunc.predict → (n,2) вероятности
        from lab4.wrap import ProbaWrapper

        with mlflow.start_run():
            mlflow.log_metric("auc", roc_auc_score(y, model.predict_proba(X)[:, 1]))
            mlflow.sklearn.log_model(ProbaWrapper(model), name="model",
                                     serialization_format="cloudpickle")
            rid = mlflow.active_run().info.run_id
        mv = mlflow.register_model(f"runs:/{rid}/model", "SmokeFraud")
        client.set_registered_model_alias("SmokeFraud", "production", mv.version)
        loaded = mlflow.pyfunc.load_model("models:/SmokeFraud@production")
        assert loaded.predict(X[:4]).shape == (4, 2), loaded.predict(X[:4]).shape
        print(f"[2/6] MLFLOW ok: version={mv.version} alias=production")

        # --- 3) FastAPI /predict (свой мини-апп на том же паттерне) -----------
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from pydantic import BaseModel

        app = FastAPI()

        @app.post("/predict")
        def predict(req: _SmokeReq):
            import numpy as _np
            return {"score": float(loaded.predict(_np.array(req.x).reshape(1, -1))[:, 1][0])}

        cl = TestClient(app)
        lat = []
        for i in range(30):
            t1 = time.perf_counter()
            r = cl.post("/predict", json={"x": [0.1] * 6})
            lat.append((time.perf_counter() - t1) * 1000)
            assert r.status_code == 200, r.text[:300]
        bad = cl.post("/predict", json={})
        assert bad.status_code == 422
        p95 = float(__import__("numpy").percentile(lat, 95))
        assert p95 < 50, f"p95={p95:.1f} ms"
        print(f"[3/6] FASTAPI ok: p95={p95:.1f} ms, 422 on bad input")

        # --- 4) дрейф ----------------------------------------------------------
        from lab4.monitor import drift_report, drift_score, check_retrain_needed

        ref = pd.DataFrame({"a": rng.normal(size=2000)})
        cur = pd.DataFrame({"a": rng.normal(loc=1.2, size=2000)})  # сдвиг всего окна:
        ds = drift_score(drift_report(ref, cur, ["a"]))            # PSI явно > 0.2
        assert ds > 0, "дрейф не замечен"
        need, why = check_retrain_needed(ds, 0.0, 0.0)
        print(f"[4/6] MONITOR ok: drift_score={ds:.2f} retrain={need} ({why})")

        # --- 5) DP -------------------------------------------------------------
        from lab4.dp import dp_curve

        curve = dp_curve(X[:300], y[:300], X[300:], y[300:], [2.0, float("inf")],
                         auc_fn=roc_auc_score, steps=300)
        assert {"epsilon_requested", "auc_test", "reachable"} <= set(curve.columns)
        print("[5/6] DP ok:", curve.to_dict("records"))

        # --- 6) часть 8 чистые функции ----------------------------------------
        from lab4.part8 import evaluate_alarms, INCIDENTS, MonitorParams, alarms_from_signals

        a0, a1 = INCIDENTS["accident_1"]
        ev = evaluate_alarms(pd.DatetimeIndex([a0 - pd.Timedelta(hours=3)]),
                             INCIDENTS["accident_1"], (a0 - pd.Timedelta(days=9), a1))
        assert ev["detected"] and ev["lead_time_h"] == 3.0
        idx = pd.date_range("2014-01-01", periods=8, freq="h")
        sig = pd.DataFrame({"prob_max": [0.9] * 8, "r1": [2.0] * 8, "y": 0.0,
                            "med1": 0.0, "q90": 1.0, "sigma": 1.0}, index=idx)
        al = alarms_from_signals(sig, MonitorParams(p_star=0.3, persist_k=2, cooldown=24))
        assert len(al) >= 1
        print(f"[6/6] PART8 ok: lead={ev['lead_time_h']}h alarms={len(al)}")

    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        import contextlib
        import gc

        gc.collect()  # закрыть sqlite-коннекты online store до unlink (Windows-локи)
        shutil.rmtree(tmp, ignore_errors=True)
        # чистим синтетический фича-пракет и стейт репозитория Feast за собой;
        # заблокированный на Windows файл не должен перекрывать итог теста
        for junk in (repo / "device_hourly_stats.parquet", repo / "data",
                     repo / "online_store.db"):
            with contextlib.suppress(PermissionError):
                if junk.is_dir():
                    shutil.rmtree(junk, ignore_errors=True)
                elif junk.exists():
                    junk.unlink()
    print(f"SMOKE OK за {time.perf_counter() - t0:.1f} c")
    return 0


if __name__ == "__main__":
    sys.exit(main())
