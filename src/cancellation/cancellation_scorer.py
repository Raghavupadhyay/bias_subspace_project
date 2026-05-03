"""
Quantifies how strong cancellation is.
"""

def cancellation_score(pos_corr: float, neg_corr: float) -> float:
    return abs(pos_corr + neg_corr)


def masking_ratio(delta_both, delta_pos, delta_neg):
    denom = delta_pos + delta_neg
    if abs(denom) < 1e-8:
        return 0.0
    return delta_both / denom