"""
metrics.py
-----------
Evaluation metrics for bias + intervention analysis.
"""

import numpy as np


def he_she_logit_gap(logits, he_id, she_id):
    last = logits[0, -1, :]
    return float(last[he_id] - last[she_id])


def kl_divergence(p_logits, q_logits):
    """KL(p || q) for last-token distributions."""
    p = p_logits[0, -1, :].softmax(-1)
    q = q_logits[0, -1, :].softmax(-1)
    return float((p * (p / (q + 1e-10)).log()).sum().item())


def masking_ratio(delta_both, individual_deltas):
    denom = sum(individual_deltas)
    if abs(denom) < 1e-8:
        return 0.0
    return float(delta_both / denom)


def summarize_intervention(results):
    deltas = np.array([r.delta_bias for r in results])
    return {
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
        "max": float(deltas.max()),
        "min": float(deltas.min()),
    }