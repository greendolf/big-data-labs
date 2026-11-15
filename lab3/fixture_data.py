"""Синтетический источник для тестов/CI (диапазоны значений соответствуют источнику)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab3.warehouse import FEATURE_COLUMNS


def synthetic_source(n: int = 5_000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    for c in FEATURE_COLUMNS:
        if c.endswith("_weight"):
            df[c] = 1.0 + rng.random(n) * 100
        elif c.endswith(("_variance", "_std", "_radius")):
            df[c] = rng.random(n) * 100
        elif c.endswith("_magnitude"):
            df[c] = 60 + rng.random(n) * 400
        elif c.endswith("_covariance"):
            df[c] = rng.normal(0, 1000, n)
        elif c.endswith("_pcc"):
            df[c] = rng.uniform(-2, 2, n)
        else:
            df[c] = rng.normal(0, 10, n)
    is_attack = rng.random(n) < 0.3
    df["Device_Name"] = [f"dev{i % 7}" for i in range(n)]
    df["Attack"] = np.where(is_attack, rng.choice(["mirai", "gafgyt"], n), "Normal")
    df["Attack_subType"] = np.where(is_attack, rng.choice(["TCP", "UDP", "DDoS"], n), "none")
    df["label"] = np.where(is_attack, 0, 1)
    return df
