from src.cancellation.pair_finder import find_cancellation_pairs, pairs_to_dataframe
import numpy as np

features = np.load("results/tables/sae_features_layer6.npy")
scores = np.load("results/tables/bias_scores.npy")

pairs = find_cancellation_pairs(features, scores)

df = pairs_to_dataframe(pairs)
df.to_csv("results/tables/exp03_cancellation_pairs.csv", index=False)