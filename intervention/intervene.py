import torch

def apply_intervention(features, direction, alpha):
    direction = torch.tensor(direction, device=features.device).float()
    return features - alpha * direction