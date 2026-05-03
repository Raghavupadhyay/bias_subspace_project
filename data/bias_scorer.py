"""
data/bias_scorer.py
--------------------
Computes ground-truth bias scores for a list of prompts using an
already-loaded model.  Score = logit(he) − logit(she) at last token.

Positive  → model predicts "he" more strongly  (male bias)
Negative  → model predicts "she" more strongly (female bias)
≈ 0       → approximately unbiased
"""

from __future__ import annotations
import numpy as np
import torch
from typing import List
from tqdm import tqdm
from transformer_lens import HookedTransformer


def score_prompts(
    model: HookedTransformer,
    prompts: List[str],
    he_token_id: int,
    she_token_id: int,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute he−she logit gap for every prompt.

    Returns
    -------
    np.ndarray of shape (N,)
    """
    scores: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring prompts"):
            batch = prompts[i : i + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)
            logits = model(tokens)               # (B, T, V)
            last   = logits[:, -1, :]            # (B, V)
            gap    = last[:, he_token_id] - last[:, she_token_id]
            scores.append(gap.cpu().numpy())

    return np.concatenate(scores)


def score_summary(scores: np.ndarray) -> dict:
    """Pretty summary statistics for a bias-score array."""
    return {
        "mean":         float(scores.mean()),
        "std":          float(scores.std()),
        "n_positive":   int((scores > 0).sum()),
        "n_negative":   int((scores < 0).sum()),
        "n_neutral":    int((scores == 0).sum()),
        "max":          float(scores.max()),
        "min":          float(scores.min()),
    }