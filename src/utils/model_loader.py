"""
src/utils/model_loader.py
--------------------------
Load GPT-2 (or any TransformerLens model) and expose helpers.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from transformer_lens import HookedTransformer


# ── Config ─────────────────────────────────────────────────────────────────────
from pathlib import Path
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    # 🔥 FIX: resolve relative to project root
    base_dir = Path(__file__).resolve().parents[2]  # go from src/utils → project root
    config_path = base_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")

    with open(config_path) as fh:
        return yaml.safe_load(fh)

# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(
    config: dict | None = None,
    config_path: str | Path = "config.yaml",
    device: str | None = None,
) -> HookedTransformer:
    """
    Load the model specified in config.

    Priority: explicit `device` arg → config["model"]["device"].
    """
    if config is None:
        config = load_config(config_path)

    model_name = config["model"]["name"]
    dev        = device or config["model"]["device"]

    print(f"Loading {model_name!r} on {dev!r} …")
    model = HookedTransformer.from_pretrained(model_name, center_unembed=True)
    model.eval()
    model.to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Parameters: {n_params:,}")
    return model


# ── Token helpers ──────────────────────────────────────────────────────────────
def get_pronoun_ids(
    model: HookedTransformer,
    config: dict | None = None,
    config_path: str | Path = "config.yaml",
) -> dict[str, int]:
    """Return token IDs for he / she / they (with leading space)."""
    if config is None:
        config = load_config(config_path)
    tok = config.get("tokens", {})
    return {
        "he":   model.to_single_token(tok.get("he",   " he")),
        "she":  model.to_single_token(tok.get("she",  " she")),
        "they": model.to_single_token(tok.get("they", " they")),
    }


# ── Backwards-compat alias ─────────────────────────────────────────────────────
get_pronoun_token_ids = get_pronoun_ids