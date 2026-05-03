"""
linear_probe.py
---------------
Ridge regression probe to extract bias direction.
"""

import numpy as np
from sklearn.linear_model import Ridge


def train_linear_probe(X: np.ndarray, y: np.ndarray, alpha=1.0):
    """
    X: (N, d_model)
    y: (N,)
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    direction = model.coef_
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    preds = model.predict(X)
    r2 = model.score(X, y)

    return {
        "direction": direction,
        "r2": float(r2),
        "preds": preds,
    }