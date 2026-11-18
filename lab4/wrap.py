"""Обёртка классификатора: predict → predict_proba.

MLflow sklearn/pyfunc по умолчанию вызывает ``.predict`` estimator'а, что для
классификатора даёт МЕТКИ класса (n,), а не вероятности. Весь продакшн-контур
(FastAPI-скоринг, canary/shadow-сверка, AUC в мониторе, batch-vs-online) работает
со СКОРОСТЬЮ положительного класса, поэтому логируем обёртку, у которой
``predict`` возвращает матрицу вероятностей (n, k). ``[:, -1]`` = положительный
класс — единственное, от чего зависят потребители.

Логается с serialization_format="cloudpickle": skops-верификация типов в MLflow 3.18
отказывается грузить кастомный класс из реестра, cloudpickle — штатный путь для
пользовательских обёрток.
"""
from __future__ import annotations

from typing import Any


class ProbaWrapper:
    """predict/predict_proba → матрица вероятностей базового классификатора."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X, y, **fit_params):  # noqa: D401
        self.estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.estimator.predict_proba(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def __getattr__(self, name: str) -> Any:   # делегировать classes_, n_features_in_ и т.п.
        # Дуни (напр. __setstate__ при unpickle) НЕ делегируем — иначе рекурсия
        # до того, как __dict__["estimator"] заполнен.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            est = object.__getattribute__(self, "__dict__")["estimator"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(est, name)
