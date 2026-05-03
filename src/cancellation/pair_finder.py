"""
pair_finder.py
--------------
THE CORE NOVEL CONTRIBUTION.

Finds "cancellation pairs" — pairs of SAE features where:
  - Feature A has positive correlation with bias scores
  - Feature B has negative correlation with bias scores
  - Their NET effect (sum of correlations) ≈ 0

The hypothesis: the model has BOTH a bias-amplifying and a bias-suppressing
feature active simultaneously. In the aggregate residual stream, they cancel.
This is why standard probing UNDER-ESTIMATES total bias.

Key metric: masking_ratio = revealed_delta / sum(individual_deltas)
If masking_ratio > 1.0, the pair is hiding more bias than visible individually.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from itertools import product as iproduct


@dataclass
class CancellationPair:
    pos_feature: int       # index of positive-bias feature
    neg_feature: int       # index of negative-bias feature
    pos_corr: float        # pearson r of pos feature with bias scores
    neg_corr: float        # pearson r of neg feature with bias scores
    net_corr: float        # pos_corr + neg_corr (near 0 = good cancellation)
    cancellation_score: float   # |net_corr| — lower is stronger cancellation
    pos_act_rate: float    # activation rate of pos feature
    neg_act_rate: float    # activation rate of neg feature
    co_activation_rate: float   # fraction of prompts where BOTH are active
    rank: int = 0


def find_cancellation_pairs(
    sae_features: np.ndarray,        # (N, n_sae_features)
    bias_scores: np.ndarray,          # (N,)
    pos_threshold: float = 0.25,
    neg_threshold: float = -0.25,
    net_threshold: float = 0.08,
    min_co_activation: float = 0.05,  # both features must co-activate enough
    top_k: int = 20,
) -> List[CancellationPair]:
    """
    Algorithm:
    1. Compute per-feature Pearson r with bias_scores
    2. Identify positive features (r > pos_threshold)
    3. Identify negative features (r < neg_threshold)
    4. For each (pos, neg) pair, compute net_corr = pos_r + neg_r
    5. Keep pairs where |net_corr| < net_threshold (strong cancellation)
    6. Filter by co-activation rate (both must fire together often enough)
    7. Return top_k pairs sorted by cancellation_score ascending

    Args:
        sae_features:       SAE feature activations, shape (N, n_features)
        bias_scores:        ground truth bias (he - she logit gap), shape (N,)
        pos_threshold:      min r to be a "positive bias feature"
        neg_threshold:      max r to be a "negative bias feature"
        net_threshold:      |pos_r + neg_r| must be below this
        min_co_activation:  pair must co-activate on at least this fraction
        top_k:              return this many pairs

    Returns:
        List of CancellationPair, sorted by cancellation_score ascending
    """
    n_features = sae_features.shape[1]

    # Step 1: compute correlations
    correlations = _batch_pearsonr(sae_features, bias_scores)
    activation_rates = (sae_features > 0).mean(axis=0)

    # Step 2-3: identify candidate features
    pos_indices = np.where(correlations > pos_threshold)[0]
    neg_indices = np.where(correlations < neg_threshold)[0]

    print(f"  Positive bias features (r > {pos_threshold}): {len(pos_indices)}")
    print(f"  Negative bias features (r < {neg_threshold}): {len(neg_indices)}")
    print(f"  Candidate pairs to evaluate: {len(pos_indices) * len(neg_indices)}")

    pairs = []

    # Step 4-6: find cancellation pairs
    for p_idx, n_idx in iproduct(pos_indices, neg_indices):
        pos_r = float(correlations[p_idx])
        neg_r = float(correlations[n_idx])
        net   = pos_r + neg_r

        if abs(net) >= net_threshold:
            continue

        # Co-activation check
        both_active = ((sae_features[:, p_idx] > 0) & (sae_features[:, n_idx] > 0))
        co_act_rate = float(both_active.mean())

        if co_act_rate < min_co_activation:
            continue

        pairs.append(CancellationPair(
            pos_feature=int(p_idx),
            neg_feature=int(n_idx),
            pos_corr=pos_r,
            neg_corr=neg_r,
            net_corr=net,
            cancellation_score=abs(net),
            pos_act_rate=float(activation_rates[p_idx]),
            neg_act_rate=float(activation_rates[n_idx]),
            co_activation_rate=co_act_rate,
        ))

    # Sort by cancellation_score (lower = stronger masking)
    pairs.sort(key=lambda x: x.cancellation_score)

    # Add ranks
    for i, p in enumerate(pairs):
        p.rank = i + 1

    print(f"  Cancellation pairs found: {len(pairs)}")
    return pairs[:top_k]


def _batch_pearsonr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized Pearson r between each column of X and vector y.
    Much faster than looping scipy.pearsonr.
    Returns array of shape (n_features,).
    """
    y = y - y.mean()
    y_std = np.std(y)
    if y_std < 1e-10:
        return np.zeros(X.shape[1])

    X_centered = X - X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0  # avoid divide by zero for dead features

    numerator = (X_centered * y[:, np.newaxis]).mean(axis=0)
    denominator = X_std * y_std
    return numerator / denominator


def pairs_to_dataframe(pairs: List[CancellationPair]) -> pd.DataFrame:
    return pd.DataFrame([vars(p) for p in pairs])


def print_pairs_summary(pairs: List[CancellationPair], top_k: int = 10):
    print(f"\nTop {min(top_k, len(pairs))} cancellation pairs:")
    print(f"{'rank':>4}  {'pos_feat':>8}  {'neg_feat':>8}  "
          f"{'pos_r':>7}  {'neg_r':>7}  {'net_r':>7}  {'co_act':>7}")
    print("-" * 62)
    for p in pairs[:top_k]:
        print(
            f"{p.rank:>4}  {p.pos_feature:>8}  {p.neg_feature:>8}  "
            f"{p.pos_corr:>7.4f}  {p.neg_corr:>7.4f}  {p.net_corr:>7.4f}  "
            f"{p.co_activation_rate:>7.4f}"
        )


def compute_aggregate_cancellation_magnitude(
    pairs: List[CancellationPair],
) -> Dict[str, float]:
    """
    Summary statistics across all found pairs.
    Used in the paper's main results table.
    """
    if not pairs:
        return {}

    pos_corrs = [p.pos_corr for p in pairs]
    neg_corrs = [p.neg_corr for p in pairs]
    net_corrs = [p.net_corr for p in pairs]
    co_acts   = [p.co_activation_rate for p in pairs]

    return {
        "n_pairs": len(pairs),
        "mean_pos_corr": float(np.mean(pos_corrs)),
        "mean_neg_corr": float(np.mean(neg_corrs)),
        "mean_net_corr": float(np.mean(np.abs(net_corrs))),
        "mean_co_activation": float(np.mean(co_acts)),
        "max_individual_signal": float(
            max(max(abs(r) for r in pos_corrs), max(abs(r) for r in neg_corrs))
        ),
        "mean_hidden_signal": float(
            np.mean([abs(p.pos_corr) + abs(p.neg_corr) for p in pairs])
        ),
    }