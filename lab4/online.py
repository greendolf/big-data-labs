"""ЛР4: сравнение batch vs online обучения (River) на потоке с дрейфом.

Честная протокольная схема:
  * поток — тестовое окно (последние 6 ч replay-сутек), отсортировано по event_time;
  * вторая половина потока помечена concept-drift фикстурой (сдвиг масштаба +
    инверсия связи у части строк) — «новая волна атак»;
  * batch-модель (production XGBoost из Registry) заморожена на train-окне;
  * online-модель (River: StandardScaler|LogisticRegression SGD) идёт потоком в
    режиме prequential: предсказала → получила метку → обновилась. Метрик «на том,
    чему учится» не считаем.

ONLINE_BATCH_DIFFERENCE = AUC_online(конец потока) − AUC_batch(тот же поток).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_stream(test: pd.DataFrame, feat_cols: list[str], chunk: int = 20_000) -> pd.DataFrame:
    from lab4.data_prep import make_drift_window

    # joined-фреймы несут event_timestamp, сырые — event_time
    tcol = "event_time" if "event_time" in test.columns else "event_timestamp"
    s = test.sort_values(tcol).reset_index(drop=True)
    half = len(s) // 2
    drift_half = make_drift_window(s.iloc[half:])
    out = pd.concat([s.iloc[:half].assign(is_drifted=False), drift_half],
                    ignore_index=True)
    return out


def _iter_dicts(df: pd.DataFrame, cols: list[str]):
    arr = df[cols].to_numpy(dtype="float64")
    for row in arr:
        yield {c: float(v) for c, v in zip(cols, row)}


def online_vs_batch(
    stream: pd.DataFrame, feat_cols: list[str], batch_model,
    chunk: int = 20_000, seed: int = 5, win: int = 10_000, step: int = 2_500,
) -> dict[str, Any]:
    """Прогон потока; возвращает кривые AUC по чанкам и финальные числа."""
    from river import linear_model, metrics, optim, preprocessing, compose
    from sklearn.metrics import roc_auc_score

    online = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.05))
    auc_stream = metrics.ROCAUC()

    stream = stream.reset_index(drop=True)
    y = stream["label"].to_numpy(dtype=int)
    preds_online = np.full(len(stream), np.nan)

    for start in range(0, len(stream), chunk):
        part = stream.iloc[start:start + chunk]
        # честный prequential: предсказываем и учимся по одному событию
        # (predict_proba ДО learn на этом же x — подсматривания нет)
        for i, (x, yi) in enumerate(zip(_iter_dicts(part, feat_cols), part["label"])):
            p = float(online.predict_proba_one(x).get(True, 0.5))
            preds_online[start + i] = p
            online.learn_one(x, int(yi))
            auc_stream.update(int(yi), p)

    Xs = stream[feat_cols].astype("float32")
    p_batch = batch_model.predict(Xs)[:, -1]
    half = len(stream) // 2

    def _win_auc(p):
        return {
            "first_half": float(roc_auc_score(y[:half], p[:half])),
            "second_half_drift": float(roc_auc_score(y[half:], p[half:])),
            "full": float(roc_auc_score(y, p)),
        }

    po = np.nan_to_num(preds_online, nan=0.5)
    # кривая по потоку: скользящее окно AUC (win) со сдвигом step
    curve = []
    for end in range(min(win, len(stream)), len(stream) + 1, step):
        lo = max(0, end - win)
        if lo == end:
            continue
        curve.append({
            "t_rows": end,
            "auc_online_win": float(roc_auc_score(y[lo:end], po[lo:end])),
            "auc_batch_win": float(roc_auc_score(y[lo:end], p_batch[lo:end])),
        })
    return {
        "batch_auc": _win_auc(p_batch),
        "online_auc": _win_auc(po),
        "online_auc_progressive_final": float(auc_stream.get()),
        "difference_final": _win_auc(po)["full"] - _win_auc(p_batch)["full"],
        "curve": pd.DataFrame(curve),
        "n_stream": len(stream), "chunk": chunk,
    }
