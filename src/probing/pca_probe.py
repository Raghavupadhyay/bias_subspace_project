"""
PCA-based bias direction extraction
"""

import numpy as np
from sklearn.decomposition import PCA


def get_pca_direction(X: np.ndarray, n_components=1):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    direction = pca.components_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return {
        "direction": direction,
        "variance_explained": float(pca.explained_variance_ratio_[0])
    }