import os
import numpy as np

from src.sae.sae_loader import load_sae
from src.sae.feature_extractor import extract_features
from src.sae.feature_annotator import correlate_features
from src.utils.model_loader import load_config


def run():
    cfg = load_config()
    tables_dir = cfg["results"]["tables_dir"]

    print("Loading saved activations...")
    X = np.load(f"{tables_dir}/activations.npy")

    print("Loading bias scores...")
    scores = np.load(f"{tables_dir}/bias_scores.npy")

    print("Loading SAE...")
    sae, _ = load_sae(
        layer=cfg["model"]["target_layer"],
        release=cfg["sae"]["release"],
        device=cfg["model"]["device"],
    )

    print("Extracting features...")
    features = extract_features(sae, X)

    print("Computing correlations...")
    corrs = correlate_features(features, scores)

    # ✅ SAVE everything
    np.save(f"{tables_dir}/sae_features_layer{cfg['model']['target_layer']}.npy", features)
    np.save(f"{tables_dir}/feature_corrs.npy", corrs)

    print("✅ SAE feature extraction complete")


if __name__ == "__main__":
    run()