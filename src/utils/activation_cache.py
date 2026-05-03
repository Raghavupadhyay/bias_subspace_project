"""
src/utils/activation_cache.py
------------------------------
Extract layer-wise residual-stream activations for a list of text prompts.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List
from tqdm import tqdm
from transformer_lens import HookedTransformer


def extract_activations(
    model: HookedTransformer,
    prompts: List[str],
    layers: List[int],
    position: str = "last",   # "last" | "mean" | int
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Extract residual-stream activations at `hook_resid_post` for each layer.

    Parameters
    ----------
    model      : HookedTransformer
    prompts    : list of text strings
    layers     : which transformer layers to cache
    position   : token position — "last", "mean", or an integer index
    batch_size : prompts per forward pass

    Returns
    -------
    dict  "blocks.{L}.hook_resid_post"  →  np.ndarray (N, d_model)
    """
    hook_names = [f"blocks.{L}.hook_resid_post" for L in layers]
    store: dict[str, list[np.ndarray]] = {h: [] for h in hook_names}

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
            batch  = prompts[i : i + batch_size]
            tokens = model.to_tokens(batch, prepend_bos=True)
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names
            )
            for h in hook_names:
                act = cache[h]   # (B, T, d_model)
                if position == "last":
                    vec = act[:, -1, :]
                elif position == "mean":
                    vec = act.mean(dim=1)
                elif isinstance(position, int):
                    vec = act[:, position, :]
                else:
                    raise ValueError(f"Unknown position={position!r}")
                store[h].append(vec.cpu().numpy())

    return {h: np.vstack(v) for h, v in store.items()}


# ── Persistence helpers ─────────────────────────────────────────────────────────
def save_activations(acts: dict[str, np.ndarray], path: str) -> None:
    """Save to a .npz file (keys have '.' replaced with '_')."""
    np.savez(path, **{k.replace(".", "_"): v for k, v in acts.items()})


def load_activations(path: str) -> dict[str, np.ndarray]:
    """Load from a .npz saved by save_activations (restores '.' in keys)."""
    data = np.load(path if path.endswith(".npz") else path + ".npz")
    # restore first two dots: blocks_6_hook_resid_post → blocks.6.hook_resid_post
    out = {}
    for raw_key, arr in data.items():
        key = raw_key.replace("_", ".", 2)  # only first two underscores
        out[key] = arr
    return out