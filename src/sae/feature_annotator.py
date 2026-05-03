"""
feature_annotator.py
--------------------
Correlate SAE features with bias scores.
"""

import numpy as np


def correlate_features(features: np.ndarray, scores: np.ndarray):
    n = features.shape[1]
    corrs = np.zeros(n)

    for i in range(n):
        corrs[i] = np.corrcoef(features[:, i], scores)[0, 1]

    return corrs