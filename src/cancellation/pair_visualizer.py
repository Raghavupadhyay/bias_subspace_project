"""
pair_visualizer.py
------------------
Plot feature pair distributions.
"""

import matplotlib.pyplot as plt


def plot_pair(pos_vals, neg_vals, save_path):
    plt.scatter(pos_vals, neg_vals, alpha=0.5)
    plt.xlabel("Positive feature")
    plt.ylabel("Negative feature")
    plt.title("Cancellation Pair Activation")
    plt.savefig(save_path)
    plt.close()