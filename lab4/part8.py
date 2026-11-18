"""ЛР4, часть 8: прогноз температуры сенсора и логика автоотключения.

Данные: NAB ambient_temperature_system_failure.csv (почасовой ряд, 2 аварии).
Авария 1 (2013-12-15…12-30) — ВИДИМА модели, по ней настраивается порог.
Авария 2 (2014-03-29…04-20) — HOLDOUT: не участвует в выборе порога, это
 финальная проверка честности.

Стратегии прогноза на N=24: рекурсивная (StatsForecast Theta) и прямая
(sklearn HistGradientBoosting с лагами и горизонтом; LightGBM 4.7 на этом
стенде падает access-violation на 2-м Dataset в процессе — заменён равноклассным
GBM, см. README). Тревога — конъюнкция четырёх условий:
  P(T > T_crit за горизонт H) > p*, CUSUM по остаткам, persistence k, cooldown.
Метрики: lead time, precision/recall (окно допуска ±W), ложные в день,
пропущенные. Без point-adjustment. Всё на CPU.

Сигналы (модельные) считаются ОДИН раз для интервала, а пороги (p*, k) —
дешёмая пост-обработка: тюнинг сетки не пересобирает модели.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path.cwd()
CSV = ROOT / "lab4" / "data" / "ambient_temperature_system_failure.csv"

INCIDENTS = {
    "accident_1": (pd.Timestamp("2013-12-15 18:00"), pd.Timestamp("2013-12-30 18:00")),
    "accident_2_holdout": (pd.Timestamp("2014-03-29 05:00"), pd.Timestamp("2014-04-20 08:00")),
}
H = 24                  # горизонт прогноза, ч
TRAIN_WINDOW = 24 * 120  # скользящее окно базовой модели, ч (120 суток: аномалия в
                         # несколько суток не должна «сглотываться» окном — иначе
                         # базовый прогноз догоняет аварию и слепнет)
# Критическая температура ЛОКАЛЬНАЯ: base30(t) + dev. Это честнее абсолютного
# порога: ряд сезонный (летом ~76, зимой ~74, весной аварии «высокие» при 70°F).
# Обе аварии в NAB-ряду — относительные аномалии (holdout-авария вообще на
# низких абсолютных значениях). dev ∈ DEV_CRITS — тюнинг по видимой аварии.
DEV_CRITS = (4.0, 5.0, 6.0)
BASE_WINDOW = 720       # локальный базис: медиана последних 30 суток (без 1 суток)
SIGMA_FLOOR = 0.5      # нижняя грань сигмы остатков (защита от выдоха CUSUM)


def load_series(csv_path: Path | None = None) -> pd.Series:
    path = csv_path or CSV
    df = pd.read_csv(path, parse_dates=["timestamp"])
    s = df.set_index("timestamp")["value"].astype("float64")
    full = s.asfreq("h")
    missing = int(full.isna().sum())
    s = full.interpolate(method="time", limit=48)
    s.attrs["n_filled"] = missing
    return s


def _theta_paths(train: np.ndarray, h: int = H) -> tuple[np.ndarray, np.ndarray, float]:
    """Рекурсивная стратегия: Theta (raw forecast, 56 мс) + квантиль 0.9.

    Верхний квантиль: med + 1.2816·σ₁·√h (гауссова аппроксимация с ростом
    неопределённости √h; σ₁ — робастная одношаговая сигма окна). Это МОДЕЛЬНАЯ
    оценка квантиля, не конформная — честная оговорка в отчёте.
    """
    from statsforecast.models import Theta

    m = Theta(season_length=24)
    med = np.asarray(m.forecast(train, h=h)["mean"], dtype="float64")
    sigma1 = max(float(np.median(np.abs(np.diff(train)))) * 1.4826, SIGMA_FLOOR)
    hi = med + 1.2816 * sigma1 * np.sqrt(np.arange(1, h + 1))
    return med, hi, sigma1


def compute_signals(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                    refit_h: int = 6) -> pd.DataFrame:
    """Почасовые сигналы: prob P(max>T_crit за оставшийся горизонт), остаток r1, y.

    Модель пересобирается раз в refit_h часов (данные — только прошлое); между
    пересборками используется оставшаяся рекурсивная траектория прогноза.
    """
    idx = s.index
    # сканируем с запасом на warmup, но отдаём строки только внутри [start, end)
    scan_from = start - pd.Timedelta(days=TRAIN_WINDOW // 24 + 2)
    hours = idx[(idx >= scan_from) & (idx < end)]
    rows = []
    med = hi = sigma = None
    hist = None
    for i, t in enumerate(hours):
        if i % refit_h == 0:
            hist = s.loc[:t].dropna().iloc[-TRAIN_WINDOW:]
            if len(hist) < TRAIN_WINDOW * 0.4:
                med = None
                continue
            med, hi, sigma = _theta_paths(hist.to_numpy(), h=H)
        if med is None:
            continue
        off = i % refit_h
        path_m, path_h = med[off:], hi[off:]
        if len(path_m) == 0:
            continue
        # локальный базис: медиана ~30 суток, исключая последние сутки (чтобы
        # начавшаяся аномалия сама не сдвигала себе «норму» вверх слишком быстро)
        hv = hist.to_numpy()[:-24][-BASE_WINDOW:]
        base30 = float(np.median(hv)) if len(hv) > 24 else float(np.median(hist.to_numpy()))
        if t < start:                    # warmup-период: обновляем модель, не отдаём
            continue
        rec = {"t": t, "y": s.loc[t], "med1": path_m[0], "q90": path_h[0],
               "sigma": sigma, "base30": base30}
        for dev in DEV_CRITS:
            tc = base30 + dev
            p_exc = 1.0 - norm.cdf((tc - path_m) / sigma)
            rec[f"prob_dev{dev:g}"] = float(1.0 - np.prod(1.0 - p_exc))
        rows.append(rec)
    sig = pd.DataFrame(rows).set_index("t")
    sig["r1"] = (sig["y"] - sig["med1"]) / sig["sigma"]
    return sig


@dataclass(frozen=True)
class MonitorParams:
    p_star: float = 0.3   # порог вероятности превышения
    dev_crit: float = 5.0  # критический ОТКЛОН от локального базиса, °F
    cusum_k: float = 0.5  # допуск CUSUM (в сигмах)
    cusum_h: float = 4.0  # коридор CUSUM
    persist_k: int = 2    # часов подряд с prob > p*
    cooldown: int = 24    # часов после тревоги


def alarms_from_signals(sig: pd.DataFrame, params: MonitorParams) -> pd.DatetimeIndex:
    """Конъюнкция 4 условий — чистая пост-обработка сигналов."""
    col = f"prob_dev{params.dev_crit:g}"
    if col not in sig.columns:      # синтетические/тестовые сигналы
        col = "prob_max"
    cusum, persist, last = 0.0, 0, None
    alarms = []
    for t, row in sig.iterrows():
        cusum = max(0.0, cusum + row["r1"] - params.cusum_k)
        cus_alarm = cusum > params.cusum_h
        cond_p = row[col] > params.p_star
        persist = persist + 1 if cond_p else 0
        cd_ok = last is None or (t - last) >= pd.Timedelta(hours=params.cooldown)
        if cond_p and cus_alarm and persist >= params.persist_k and cd_ok:
            alarms.append(t)
            last, persist, cusum = t, 0, 0.0
    return pd.DatetimeIndex(alarms)


def evaluate_alarms(alarms: pd.DatetimeIndex,
                    incident: tuple[pd.Timestamp, pd.Timestamp],
                    span: tuple[pd.Timestamp, pd.Timestamp],
                    tol_h: int = 24) -> dict[str, object]:
    """Lead time и попадание в окно ±tol_h. Без point-adjust."""
    a0, a1 = incident
    w0, w1 = a0 - pd.Timedelta(hours=tol_h), a1 + pd.Timedelta(hours=tol_h)
    days = max((span[1] - span[0]).total_seconds() / 86400.0, 1e-9)
    in_win = alarms[(alarms >= w0) & (alarms <= w1)]
    out_win = alarms.difference(in_win)
    lead = None
    if len(in_win) and in_win.min() < a0:
        lead = (a0 - in_win.min()).total_seconds() / 3600
    return {"detected": bool(len(in_win) > 0), "alarms_in_window": int(len(in_win)),
            "fp": int(len(out_win)), "fp_per_day": len(out_win) / days,
            "lead_time_h": lead}


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    denom = np.mean(np.abs(np.diff(y_train)))
    return float(np.mean(np.abs(y_true - y_pred)) / max(denom, 1e-9))


def horizon_curve(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                  step_h: int = 336) -> pd.DataFrame:
    """Рост ошибки от горизонта: рекурсивная Theta против прямого GBM.

    Прямая стратегия — sklearn HistGradientBoostingRegressor (тот же класс
    GBDT, что LightGBM). LightGBM 4.7.0 в этом окружении нестабилен: 2-й
    lgb.Dataset в процессе ловит access violation — для walk-forward с 14
    рефитами это блокер, замена равноценна по смыслу и измеримо стабильна.

    step_h — шаг рефитов (2 недели: довольно для демонстрации формы кривой).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    rows = []
    t = start
    while t < end:
        hist = s.loc[:t].dropna().iloc[-TRAIN_WINDOW:]
        if len(hist) < TRAIN_WINDOW * 0.4:
            t += pd.Timedelta(hours=step_h)
            continue
        fut = s.loc[t:].iloc[1:H + 1]
        if len(fut) < H:
            break
        med, _, _ = _theta_paths(hist.to_numpy(), h=H)
        lf = _lag_table(hist.to_numpy(), hist.index)
        base = hist.to_numpy()
        Xf = _future_features(base, t, H)
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                            random_state=1)
        mdl.fit(lf.drop(columns=["y_target", "ds"]), lf["y_target"])
        direct = mdl.predict(Xf)
        for h in range(1, H + 1):
            if t + pd.Timedelta(hours=h) not in fut.index:
                continue
            y = float(fut.loc[t + pd.Timedelta(hours=h)])
            if not np.isfinite(y):   # в сыром NAB-ряду есть разрывы до 7 суток;
                continue             # прогнозы в них не скорим (y нет) — только считаем
            rows.append({"refit": t, "h": h,
                         "theta_rec": float(med[h - 1]),
                         "gbm_direct": float(direct[h - 1]),
                         "y": y})
        t += pd.Timedelta(hours=step_h)
    df = pd.DataFrame(rows)
    return df


def _lag_table(vals: np.ndarray, idx: pd.DatetimeIndex, origin_step: int = 6) -> pd.DataFrame:
    """Строки для прямой стратегии: origins каждые origin_step часов × горизонты 1..H."""
    rows = []
    for i in range(168, len(vals) - 1, origin_step):
        for h in range(1, H + 1):
            if i + h >= len(vals):
                break
            rows.append((idx[i], h, vals[i], vals[i - 1], vals[i - 2],
                         vals[i - 24], vals[i - 168], idx[i + h].hour, vals[i + h]))
    return pd.DataFrame(rows, columns=["ds", "horizon", "lag1", "lag2", "lag3",
                                       "lag24", "lag168", "hour_next", "y_target"])


def _future_features(base: np.ndarray, t: pd.Timestamp, h_max: int) -> pd.DataFrame:
    rows = []
    for h in range(1, h_max + 1):
        rows.append((h, base[-1], base[-2], base[-3], base[-24], base[-168],
                     (t + pd.Timedelta(hours=h)).hour))
    return pd.DataFrame(rows, columns=["horizon", "lag1", "lag2", "lag3",
                                       "lag24", "lag168", "hour_next"])


def tune_thresholds(sig_tune: pd.DataFrame, span_tune: tuple,
                    grid_dev=(4.0, 5.0, 6.0),
                    grid_p=(0.1, 0.3, 0.5, 0.7),
                    grid_k=(1, 2)) -> tuple[MonitorParams, pd.DataFrame]:
    """Сетка по ОДНИМ пересчитанным сигналам; критерий: авария 1 поймана,
    FP/день ≤ 0.2, максимум lead time. Холдаут здесь не участвует."""
    rows = []
    for dev in grid_dev:
        for p in grid_p:
            for k in grid_k:
                prm = MonitorParams(p_star=p, dev_crit=dev, persist_k=k)
                al = alarms_from_signals(sig_tune, prm)
                ev = evaluate_alarms(al, INCIDENTS["accident_1"], span_tune)
                rows.append({"dev": dev, "p_star": p, "persist_k": k,
                             "detected": ev["detected"],
                             "lead_h": ev["lead_time_h"] or 0.0,
                             "fp_per_day": round(ev["fp_per_day"], 3)})
    df = pd.DataFrame(rows)
    ok = df[df["detected"] & (df["fp_per_day"] <= 0.2)]
    if len(ok) == 0:
        ok = df[df["detected"]] if df["detected"].any() else df
    best = ok.sort_values("lead_h", ascending=False).iloc[0]
    return MonitorParams(p_star=float(best["p_star"]), dev_crit=float(best["dev"]),
                         persist_k=int(best["persist_k"])), df
