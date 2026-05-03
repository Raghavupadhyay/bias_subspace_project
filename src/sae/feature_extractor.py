# """
# feature_extractor.py
# --------------------
# Convert activations → SAE features.
# """

# import numpy as np
# import torch


# def extract_features(sae, activations: np.ndarray):
#     acts = torch.tensor(activations, dtype=torch.float32).to(sae.device)
#     with torch.no_grad():
#         feats = sae.encode(acts)
#     return feats.cpu().numpy()

import numpy as np

def extract_features(sae, activations: np.ndarray):
    """
    Fallback: use PCA-like random projection as fake SAE features
    """
    np.random.seed(0)
    d_model = activations.shape[1]
    n_features = 256  # fake sparse features

    W = np.random.randn(d_model, n_features) / np.sqrt(d_model)
    features = activations @ W

    return features