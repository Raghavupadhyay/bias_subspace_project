"""
resample_ablation.py
--------------------
Resample-based causal baseline.
"""

import torch


def resample_feature(features, idx):
    perm = torch.randperm(features.shape[0])
    features[:, idx] = features[perm, idx]
    return features