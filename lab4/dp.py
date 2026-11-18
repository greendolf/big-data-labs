"""ЛР4: дифференциальная приватность при обучении (уникальное требование, вариант 4).

DP-SGD для логистической регрессии на NumPy:
  * покомпонентное усечение градиентов (clip по L2-норме C),
  * гауссов шум на сумму усечённых градиентов (std σ·C на шаг),
  * учёт бюджета через RDP-accountant без claims об усилении выборок
    (консервативная полная композиция — защищаемая, а не «красивая» оценка).

Гарантия: обучение (X_train, y_train) → (w, b) удовлетворяет (ε, δ)-DP для δ < 1/n
(по умолчанию 1e-6) при шуме σ, подобранном под целевой ε. Нормировка признаков —
фиксированный аугментатор (см. оговорку в ноутбуке: агрегированная сетевая
телеметрия, не персональные атрибуты; основная защита — на этапе обучения).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ALPHAS = np.array([1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])


def rdp_epsilon(sigma: float, steps: int, delta: float) -> float:
    """ε от (σ, T, δ): гауссов механизм, полная композиция, минимум по α."""
    if sigma <= 0:
        return float("inf")
    rdp = _ALPHAS / (2.0 * sigma * sigma) * steps
    eps = rdp + np.log(1.0 / delta) / (_ALPHAS - 1.0)
    return float(eps.min())


_SIGMA_MAX = 100.0  # потолок шума бинарного поиска: практичнее «заглушить модель» нет смысла


def epsilon_min(steps: int, delta: float, sigma_max: float = _SIGMA_MAX) -> float:
    """Наименьший достижимый ε при полной композиции (без усиления выборок).

    Меньше него бюджет недостижим без аккаунтера с усилением Пуассоновской выборки
    (Opacus/TF-Privacy) — честная граница нашей консервативной оценки.
    """
    return rdp_epsilon(sigma_max, steps, delta)


def sigma_for_epsilon(target: float, steps: int, delta: float) -> float:
    """Меньший σ, дающий ε <= target (бинарный поиск по монотонной ε(σ))."""
    lo, hi = 1e-3, _SIGMA_MAX
    if rdp_epsilon(hi, steps, delta) > target:
        raise ValueError(
            f"ε={target} недостижим при полной композиции (минимум {epsilon_min(steps, delta):.2f}); "
            f"нужен аккаунтер с усилением выборок")
    for _ in range(60):
        mid = (lo + hi) / 2
        if rdp_epsilon(mid, steps, delta) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class DPResult:
    epsilon: float
    delta: float
    sigma: float
    steps: int
    auc: float
    weights: np.ndarray
    bias: float


def train_logreg_dp(
    X: np.ndarray, y: np.ndarray, *, steps: int = 3000, batch: int = 1024,
    lr: float = 0.08, clip_c: float = 1.0, sigma: float = 0.0,
    epsilon: float | None = None, delta: float = 1e-6, seed: int = 42,
    eval_fn=None,
) -> DPResult:
    """SGD логистической регрессии; sigma=0 и epsilon=None — baseline без приватности.

    При epsilon заданном и sigma==0 — σ подбирается под бюджет (полная композиция:
    steps шагов без усиления выборок).
    """
    rng = np.random.default_rng(seed)
    n, d0 = X.shape
    if sigma <= 0 and epsilon is not None and np.isfinite(epsilon):
        sigma = sigma_for_epsilon(epsilon, steps, delta)

    # bias = вес фиктивной фичи ≡ 1: общий lr, общий per-sample clip
    # (нескейлированный bias-шаг давал positive-feedback runaway в ранней версии)
    Xa = np.hstack([np.asarray(X, dtype="float64"), np.ones((n, 1))])
    d = d0 + 1
    w = np.zeros(d)
    for t in range(steps):
        idx = rng.integers(0, n, batch)
        xb, yb = Xa[idx], y[idx]
        err = _sigmoid(xb @ w) - yb
        if sigma > 0:
            # per-sample L2-clip: ||x_i * err_i||_2 = |err_i| * ||x_i||
            norms = np.abs(err) * np.linalg.norm(xb, axis=1)
            scale = np.minimum(1.0, clip_c / np.maximum(norms, 1e-12))
            g = xb.T @ (err * scale) / batch
            g = g + rng.normal(0.0, sigma * clip_c / batch, size=d)
        else:
            g = xb.T @ err / batch
        w -= lr * g

    auc = eval_fn(y, _sigmoid(Xa @ w)) if eval_fn is not None else float("nan")
    return DPResult(
        epsilon=(rdp_epsilon(sigma, steps, delta) if sigma > 0 else float("inf")),
        delta=delta, sigma=sigma, steps=steps, auc=float(auc),
        weights=w[:d0], bias=float(w[d0]),
    )


def predict_proba(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    return _sigmoid(X @ w + b)


def dp_curve(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray,
    epsilons: list[float], *, auc_fn, strict: bool = True, **train_kw,
) -> pd.DataFrame:
    """Кривая «цена приватности»: AUC на тесте по каждому ε (∞ = baseline).

    strict=False: недостижимые при полной композиции ε (меньше epsilon_min) помечаются
    reachable=False с AUC=NaN — граница честной оценки, а не сбой.
    """
    steps = train_kw.get("steps", 3000)
    delta = train_kw.get("delta", 1e-6)
    rows = []
    for eps in epsilons:
        try:
            res = train_logreg_dp(x_tr, y_tr, epsilon=eps,
                                  eval_fn=lambda yy, p: auc_fn(yy, p), **train_kw)
        except ValueError:
            if strict:
                raise
            rows.append({"epsilon_requested": eps, "epsilon_spent": float("inf"),
                         "sigma": float("nan"), "auc_train_eval": float("nan"),
                         "auc_test": float("nan"), "reachable": False})
            continue
        auc_te = auc_fn(y_te, predict_proba(res.weights, res.bias, x_te))
        rows.append({"epsilon_requested": eps, "epsilon_spent": round(res.epsilon, 4),
                     "sigma": round(res.sigma, 4), "auc_train_eval": round(res.auc, 4),
                     "auc_test": round(auc_te, 4), "reachable": True})
    return pd.DataFrame(rows)
