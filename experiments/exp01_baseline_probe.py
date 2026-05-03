import os
import numpy as np

from src.utils.model_loader import load_model, get_pronoun_token_ids, load_config
from src.utils.activation_cache import extract_activations
from src.probing.linear_probe import train_linear_probe
from data.prompts_winobias import generate_prompts
from data.bias_scorer import score_prompts


def run():
    cfg = load_config()

    tables_dir = cfg["results"]["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(cfg)

    token_ids = get_pronoun_token_ids(model, cfg)
    he_id, she_id = token_ids["he"], token_ids["she"]

    print("Generating prompts...")
    prompts_obj = generate_prompts()
    prompts = [p.text for p in prompts_obj]

    target_layer = cfg["model"]["target_layer"]

    print("Extracting activations...")
    acts = extract_activations(model, prompts, [target_layer])
    X = list(acts.values())[0]

    # ✅ SAVE activations
    np.save(f"{tables_dir}/activations.npy", X)

    print("Scoring bias...")
    scores = score_prompts(model, prompts, he_id, she_id)

    # ✅ SAVE scores
    np.save(f"{tables_dir}/bias_scores.npy", scores)

    print("Training probe...")
    probe = train_linear_probe(X, scores)

    # ✅ SAVE direction
    np.save(f"{tables_dir}/direction_layer{target_layer}.npy", probe["direction"])

    print("✅ Baseline probe complete")


if __name__ == "__main__":
    run()