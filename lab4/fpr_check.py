"""Замер FPR продакшн-модели на тестовой матрице (требование задания: FPR < 1%).

Ноутбук фиксирует AUC/latency/PSI; FPR — отдельное воспроизводимое измерение
той же модели, что под алиасом `production` в реестре (рандомный лес, v1).
Порог 0.5 — дефолт сервинга. Матрица собирается детерминированно теми же
seed'ами, что и в ноутбуке (lab4.mlpipeline.build_training_matrix).

Запуск:  python -X utf8 -m lab4.fpr_check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lab4 import mlpipeline as mp


def main() -> None:
    mp.setup_mlflow()
    mt = mp.build_training_matrix(sample_train=200_000, sample_test=250_000)
    import mlflow
    model = mlflow.pyfunc.load_model(f"models:/{mp.MODEL_NAME}@production")
    p = np.asarray(model.predict(mt["X_test"]))[:, -1]
    y = mt["y_test"]
    pred = (p >= 0.5).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 1) != (y == 1)).sum())
    fpr = fp / max(fp + tn, 1)
    print(f"auc={roc_auc_score(y, p):.6f} threshold=0.5")
    print(f"FP={fp} TN={tn} FN+FP(ошибок)={fn} n={len(y)}")
    print(f"FPR={fpr:.6f} ({fpr:.3%})  требование < 1%: "
          f"{'ВЫПОЛНЕНО' if fpr < 0.01 else 'НЕ ВЫПОЛНЕНО'}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
