"""
ablation.py — FINAL FIXED VERSION
Supports BOTH:
1. Real SAE
2. Fallback (no SAE → random feature directions)
"""

import numpy as np
import torch
from typing import List, Dict
from dataclasses import dataclass, field
from tqdm import tqdm
from transformer_lens import HookedTransformer


@dataclass
class AblationResult:
    prompt: str
    bias_before: float
    bias_after: float
    delta_bias: float
    top_prob_before: float
    top_prob_after: float
    intervention_type: str
    alpha: float = 1.0
    feature_indices: List[int] = field(default_factory=list)


# ─────────────────────────────────────────────
# Direction ablation
# ─────────────────────────────────────────────
def direction_ablation(
    model: HookedTransformer,
    prompts: List[str],
    direction: np.ndarray,
    layer: int,
    he_id: int,
    she_id: int,
    alpha: float = 1.0,
):
    dir_tensor = torch.tensor(direction, dtype=torch.float32).to(model.cfg.device)
    results = []

    for prompt in tqdm(prompts, desc=f"Direction ablation α={alpha}"):
        before = _get_logit_gap(model, prompt, he_id, she_id)
        top_before = _get_top_prob(model, prompt)

        def hook_fn(act, hook):
            proj = (act[:, -1, :] @ dir_tensor).unsqueeze(-1) * dir_tensor
            act[:, -1, :] -= alpha * proj / (dir_tensor @ dir_tensor)
            return act

        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
        )

        after = _logit_gap_from_logits(logits, he_id, she_id)
        top_after = _top_prob_from_logits(logits)

        results.append(AblationResult(
            prompt, before, after, before - after,
            top_before, top_after, "direction", alpha
        ))

    return results


# ─────────────────────────────────────────────
# Feature ablation (FIXED)
# ─────────────────────────────────────────────
def feature_ablation(
    model: HookedTransformer,
    sae,
    prompts: List[str],
    feature_indices: List[int],
    layer: int,
    he_id: int,
    she_id: int,
    intervention_type="single_feature",
):
    device = model.cfg.device
    results = []

    for prompt in tqdm(prompts, desc=f"Feature ablation {feature_indices[:2]}"):
        before = _get_logit_gap(model, prompt, he_id, she_id)
        top_before = _get_top_prob(model, prompt)

        def hook_fn(act, hook):
            last_act = act[:, -1, :]

            # 🔥 CASE 1: REAL SAE
            if sae is not None:
                feat_idx = torch.tensor(feature_indices, dtype=torch.long).to(device)
                feats = sae.encode(last_act)
                feats[:, feat_idx] = 0.0
                recon = sae.decode(feats)
                act[:, -1, :] = recon

            # 🔥 CASE 2: FALLBACK (NO SAE)
            else:
                vec = last_act.clone()

                for idx in feature_indices:
                    torch.manual_seed(int(idx))  # deterministic
                    direction = torch.randn_like(vec)
                    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

                    proj = (vec * direction).sum(dim=-1, keepdim=True) * direction
                    vec = vec - proj  # remove that component

                act[:, -1, :] = vec

            return act

        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
        )

        after = _logit_gap_from_logits(logits, he_id, she_id)
        top_after = _top_prob_from_logits(logits)

        results.append(AblationResult(
            prompt, before, after, before - after,
            top_before, top_after, intervention_type,
            feature_indices=feature_indices
        ))

    return results


# ─────────────────────────────────────────────
# Pair experiment
# ─────────────────────────────────────────────
def pair_ablation_experiment(
    model,
    sae,
    prompts,
    pos_feature,
    neg_feature,
    layer,
    he_id,
    she_id,
):
    pos_feature = int(pos_feature)
    neg_feature = int(neg_feature)

    return {
        "pos_only": feature_ablation(model, sae, prompts, [pos_feature], layer, he_id, she_id, "pos_only"),
        "neg_only": feature_ablation(model, sae, prompts, [neg_feature], layer, he_id, she_id, "neg_only"),
        "both": feature_ablation(model, sae, prompts, [pos_feature, neg_feature], layer, he_id, she_id, "pair_both"),
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _get_logit_gap(model, prompt, he_id, she_id):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    logits = model(tokens)
    return float(logits[0, -1, he_id] - logits[0, -1, she_id])


def _logit_gap_from_logits(logits, he_id, she_id):
    return float(logits[0, -1, he_id] - logits[0, -1, she_id])


def _get_top_prob(model, prompt):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    logits = model(tokens)
    return float(logits[0, -1, :].softmax(-1).max())


def _top_prob_from_logits(logits):
    return float(logits[0, -1, :].softmax(-1).max())


def aggregate_results(results):
    """
    Compute full summary (FIXED to match exp04 expectations)
    """
    deltas = np.array([r.delta_bias for r in results])
    bias_before = np.array([r.bias_before for r in results])
    bias_after = np.array([r.bias_after for r in results])
    top_before = np.array([r.top_prob_before for r in results])
    top_after = np.array([r.top_prob_after for r in results])

    return {
        "mean_delta_bias": float(deltas.mean()),
        "std_delta_bias": float(deltas.std()),
        "mean_bias_before": float(bias_before.mean()),
        "mean_bias_after": float(bias_after.mean()),
        "mean_top_prob_before": float(top_before.mean()),
        "mean_top_prob_after": float(top_after.mean()),
        "capability_degradation": float((top_before - top_after).mean()),
        "n": len(results),
    }


# ─────────────────────────────────────────────
# Random control ablation (FIX)
# ─────────────────────────────────────────────
def random_pair_control(
    model: HookedTransformer,
    sae,
    prompts: List[str],
    n_sae_features: int,
    layer: int,
    he_id: int,
    she_id: int,
    n_controls: int = 20,
    rng_seed: int = 42,
):
    """
    Ablate random feature pairs (negative control).
    Expected: delta_bias ≈ 0
    """
    rng = np.random.default_rng(rng_seed)
    all_results = []

    for _ in range(n_controls):
        idx_a, idx_b = rng.choice(n_sae_features, size=2, replace=False).tolist()

        results = feature_ablation(
            model,
            sae,
            prompts[:20],  # small subset for speed
            [int(idx_a), int(idx_b)],
            layer,
            he_id,
            she_id,
            intervention_type="random_control",
        )

        all_results.extend(results)

    return all_results