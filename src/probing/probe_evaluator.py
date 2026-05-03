"""
probe_evaluator.py
------------------
Evaluate alignment between probe predictions and bias scores.
"""

import numpy as np


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_probe(preds, targets):
    return {
        "corr": correlation(preds, targets),
        "mse": float(((preds - targets) ** 2).mean()),
    }