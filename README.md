# Bias Cancellation in Language Models
## Hidden Bias via Opposing Sparse Features: A Causal Subspace Analysis

---

## Research Hypothesis

> Social bias in language models is not merely *distributed* — it is actively *masked* by opposing sparse features within the same layer. We demonstrate this using SAE decomposition, where we identify feature pairs with opposing bias contributions whose sum approaches zero, yet whose individual ablation reveals hidden bias signal.

---

## Project Structure

```
bias_cancellation_project/
│
├── data/
│   ├── prompts_winobias.py          # WinoBias-style controlled prompts
│   ├── prompts_custom.py            # Custom profession prompts
│   └── bias_scorer.py               # Ground truth bias score assignment
│
├── src/
│   ├── utils/
│   │   ├── model_loader.py          # GPT-2 loading via TransformerLens
│   │   ├── activation_cache.py      # Layer-wise activation extraction
│   │   └── metrics.py               # he/she logit gap, KL divergence
│   │
│   ├── probing/
│   │   ├── linear_probe.py          # Ridge regression bias direction
│   │   ├── pca_probe.py             # PCA-based direction extraction
│   │   └── probe_evaluator.py       # Probe accuracy, R², direction alignment
│   │
│   ├── sae/
│   │   ├── sae_loader.py            # Load pretrained SAE (sae-lens / EleutherAI)
│   │   ├── feature_extractor.py     # Encode activations → SAE features
│   │   └── feature_annotator.py     # Correlate features with bias scores
│   │
│   ├── cancellation/
│   │   ├── pair_finder.py           # Find opposing feature pairs (CORE NOVELTY)
│   │   ├── cancellation_scorer.py   # Quantify masking magnitude
│   │   └── pair_visualizer.py       # Plot feature pair distributions
│   │
│   └── causal/
│       ├── ablation.py              # Activation patching + feature ablation
│       ├── resample_ablation.py     # Resample-based ablation (causal baseline)
│       ├── path_patch.py            # Path patching for stronger causality
│       └── intervention_eval.py     # Measure delta_bias, capability degradation
│
├── experiments/
│   ├── exp01_baseline_probe.py      # Reproduce linear probe baseline
│   ├── exp02_sae_features.py        # SAE feature correlation with bias
│   ├── exp03_cancellation_pairs.py  # Find and validate cancellation pairs
│   ├── exp04_causal_ablation.py     # Ablate pairs, measure output shift
│   ├── exp05_layer_sweep.py         # Repeat across layers 0–11
│   └── exp06_full_pipeline.py       # End-to-end result generation
│
├── results/
│   ├── figures/                     # Auto-generated plots
│   └── tables/                      # CSV result tables
│
├── tests/
│   ├── test_pair_finder.py
│   ├── test_ablation.py
│   └── test_metrics.py
│
├── requirements.txt
├── config.yaml
└── run_all.sh
```

---

## Experimental Pipeline

```
Step 1: Load GPT-2 small via TransformerLens
Step 2: Run WinoBias + custom prompts → extract residual stream at layers [0,3,6,9,11]
Step 3: Assign bias scores (he/she logit gap as ground truth)
Step 4: Train linear probe → get baseline direction + delta_bias
Step 5: Load pretrained SAE (sae-lens, GPT-2 small, matching layers)
Step 6: Encode activations → SAE features (sparse, named)
Step 7: Correlate each SAE feature with bias scores
Step 8: Find CANCELLATION PAIRS — feature i (pos) + feature j (neg) ≈ 0
Step 9: Ablate cancellation pairs → measure revealed delta_bias
Step 10: Compare: pair ablation delta > direction ablation delta? (main claim)
Step 11: Control: ablate random pairs → delta ≈ 0 (validates specificity)
Step 12: Layer sweep → which layer has most cancellation?
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| `delta_bias` | Change in he/she logit gap after intervention |
| `cancellation_score` | `abs(pos_corr + neg_corr)` — lower = more hidden |
| `revealed_bias` | delta_bias when BOTH features ablated vs neither |
| `masking_ratio` | revealed_bias / (sum of individual ablation deltas) |
| `capability_degradation` | Top-1 token probability drop after ablation |

---

## Expected Results (Hypotheses to Test)

**H1**: Cancellation pairs exist — SAE features with `corr > 0.3` and `corr < -0.3` whose sum is near zero.

**H2**: Ablating cancellation pairs reveals MORE bias than ablating the linear probe direction alone.

**H3**: Ablating only ONE member of a pair reveals LESS bias than ablating both (asymmetry test).

**H4**: Random pair ablation has near-zero delta_bias (negative control).

**H5**: Cancellation density increases in middle layers (6–9) — where contextual processing peaks.

---

