"""
exp04_causal_ablation.py
------------------------
Experiment 4: THE MAIN RESULT.

For each top cancellation pair:
  Condition A: ablate pos feature only        → delta_A
  Condition B: ablate neg feature only        → delta_B
  Condition C: ablate BOTH                    → delta_C
  Condition D: ablate random pair (control)   → delta_D ≈ 0

PREDICTION:
  delta_C > delta_A + delta_B  →  masking_ratio > 1.0
  delta_D ≈ 0                  →  effect is specific to bias features

This is Table 1 in the paper.
Prerequisite: exp01 + exp02 + exp03.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.model_loader import load_model, load_config, get_pronoun_token_ids
from src.sae.sae_loader import load_sae
from src.causal.ablation import (
    direction_ablation,
    pair_ablation_experiment,
    random_pair_control,
    aggregate_results,
)
from src.utils.metrics import masking_ratio, summarize_intervention
from src.cancellation.pair_finder import CancellationPair
from data.prompts_winobias import generate_prompts


def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    tables_dir = cfg["results"]["tables_dir"]
    figures_dir = cfg["results"]["figures_dir"]

    target_layer = cfg["model"]["target_layer"]

    # ── 1. Load model + SAE ───────────────────────────────────
    model = load_model(cfg)
    token_ids = get_pronoun_token_ids(model, cfg)
    he_id, she_id = token_ids["he"], token_ids["she"]

    sae, _ = load_sae(
        layer=target_layer,
        release=cfg["sae"]["release"],
        device=cfg["model"]["device"],
    )

    # ── 2. Load prompts ───────────────────────────────────────
    prompts_obj = generate_prompts()
    prompts = [p.text for p in prompts_obj]
    # Use smaller subset for speed — increase for final results
    eval_prompts = prompts[:100]
    print(f"Evaluating on {len(eval_prompts)} prompts")

    # ── 3. Load baseline direction for comparison ─────────────
    direction = np.load(f"{tables_dir}/direction_layer{target_layer}.npy")

    # ── 4. Load top cancellation pairs ────────────────────────
    pairs_df = pd.read_csv(f"{tables_dir}/exp03_cancellation_pairs.csv")
    top_pairs = [
        CancellationPair(**row)
        for _, row in pairs_df.head(5).iterrows()
    ]
    print(f"Testing top {len(top_pairs)} cancellation pairs")

    # ── 5. Baseline: direction ablation ──────────────────────
    print("\n── Baseline: direction ablation ──")
    dir_results = direction_ablation(
        model, eval_prompts, direction, target_layer, he_id, she_id, alpha=1.0
    )
    dir_summary = aggregate_results(dir_results)
    print(f"  Direction ablation: delta_bias={dir_summary['mean_delta_bias']:.4f}")

    # ── 6. Pair ablation experiment ───────────────────────────
    all_rows = []

    for pair in top_pairs:
        print(f"\n── Pair (pos={pair.pos_feature}, neg={pair.neg_feature}) ──")
        print(f"   pos_r={pair.pos_corr:.4f}, neg_r={pair.neg_corr:.4f}, net_r={pair.net_corr:.4f}")

        conditions = pair_ablation_experiment(
            model, sae, eval_prompts,
            pair.pos_feature, pair.neg_feature,
            target_layer, he_id, she_id,
        )

        delta_pos  = aggregate_results(conditions["pos_only"])["mean_delta_bias"]
        delta_neg  = aggregate_results(conditions["neg_only"])["mean_delta_bias"]
        delta_both = aggregate_results(conditions["both"])["mean_delta_bias"]
        mr = masking_ratio(delta_both, [delta_pos, delta_neg])

        cap_both = aggregate_results(conditions["both"])["capability_degradation"]

        print(f"   delta_pos={delta_pos:.4f}, delta_neg={delta_neg:.4f}, delta_both={delta_both:.4f}")
        print(f"   masking_ratio={mr:.4f}  ({'✓ CANCELLATION' if mr > 1.0 else '✗ no cancellation'})")
        print(f"   capability_degradation={cap_both:.4f}")

        all_rows.append({
            "pos_feature": pair.pos_feature,
            "neg_feature": pair.neg_feature,
            "pos_corr": pair.pos_corr,
            "neg_corr": pair.neg_corr,
            "net_corr": pair.net_corr,
            "delta_pos_only": delta_pos,
            "delta_neg_only": delta_neg,
            "delta_both": delta_both,
            "masking_ratio": mr,
            "direction_delta": dir_summary["mean_delta_bias"],
            "capability_degradation": cap_both,
            "cancellation_confirmed": mr > 1.0,
        })

    # ── 7. Random pair controls ───────────────────────────────
    print("\n── Random pair control ──")
    n_sae = np.load(f"{tables_dir}/sae_features_layer{target_layer}.npy").shape[1]
    ctrl_results = random_pair_control(
        model, sae, eval_prompts, n_sae, target_layer, he_id, she_id,
        n_controls=cfg["ablation"]["control_pairs"],
    )
    ctrl_summary = aggregate_results(ctrl_results)
    print(f"  Random control: delta_bias={ctrl_summary['mean_delta_bias']:.4f} "
          f"(expected ≈ 0)")

    # ── 8. Save and plot ──────────────────────────────────────
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(f"{tables_dir}/exp04_ablation_results.csv", index=False)

    _plot_main_results(results_df, dir_summary, ctrl_summary, figures_dir, target_layer)

    print(f"\n✓ Experiment 4 complete.")
    print(f"  Pairs with masking_ratio > 1.0: "
          f"{results_df['cancellation_confirmed'].sum()} / {len(results_df)}")
    return results_df


def _plot_main_results(results_df, dir_summary, ctrl_summary, figures_dir, layer):
    """
    Main results figure — this is the key plot for the paper.
    Shows delta_bias for: direction | pos_only | neg_only | both | control
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: delta_bias comparison per pair ──────────────────
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.2

    ax.bar(x - width*1.5, results_df["delta_pos_only"], width, label="Pos only", color="#F0997B", alpha=0.9)
    ax.bar(x - width*0.5, results_df["delta_neg_only"], width, label="Neg only", color="#85B7EB", alpha=0.9)
    ax.bar(x + width*0.5, results_df["delta_both"],     width, label="Both (pair)", color="#1D9E75", alpha=0.9)
    ax.axhline(dir_summary["mean_delta_bias"], color="#534AB7", linestyle="--", linewidth=1.5, label="Direction probe")
    ax.axhline(ctrl_summary["mean_delta_bias"], color="#888780", linestyle=":", linewidth=1.5, label="Random control")

    ax.set_xlabel("Cancellation pair rank")
    ax.set_ylabel("Mean Δ bias (he-she logit gap)")
    ax.set_title(f"Delta bias by ablation condition — layer {layer}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pair {i+1}" for i in x])
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)

    # ── Right: masking ratio ──────────────────────────────────
    ax = axes[1]
    colors = ["#1D9E75" if v > 1.0 else "#D85A30" for v in results_df["masking_ratio"]]
    bars = ax.bar(x, results_df["masking_ratio"], color=colors, alpha=0.9)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="Threshold (ratio=1)")
    ax.set_xlabel("Cancellation pair rank")
    ax.set_ylabel("Masking ratio (δ_both / Σδ_individual)")
    ax.set_title("Masking ratio — values > 1.0 confirm cancellation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Pair {i+1}" for i in x])
    ax.legend()

    # Annotate bars
    for bar, v in zip(bars, results_df["masking_ratio"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/exp04_main_results_layer{layer}.png", dpi=150)
    plt.close()
    print(f"  Saved main results plot")


if __name__ == "__main__":
    run()