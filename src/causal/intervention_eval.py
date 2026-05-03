"""
intervention_eval.py
--------------------
Compute final evaluation metrics.
"""

def delta_bias(before, after):
    return before - after


def capability_drop(before, after):
    return before - after