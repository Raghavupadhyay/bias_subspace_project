# """
# sae_loader.py
# -------------
# Robust SAE loader supporting multiple sae-lens versions.
# """

# def load_sae(layer: int, release: str, device: str):
#     from sae_lens import SAE

#     # 🔥 Try newest API
#     try:
#         sae = SAE.from_pretrained(
#             release=release,
#             layer=layer,
#             device=device,
#         )
#         print("✅ Loaded SAE (new API)")
#         return sae, getattr(sae, "cfg", None)

#     except TypeError:
#         pass

#     # 🔥 Try mid API (no layer)
#     try:
#         sae = SAE.from_pretrained(
#             release=release,
#             device=device,
#         )
#         print("⚠️ Loaded SAE (no layer API)")
#         return sae, getattr(sae, "cfg", None)

#     except TypeError:
#         pass

#     # 🔥 Try OLD API (requires sae_id)
#     try:
#         sae_id = f"{release}_layer_{layer}"
#         sae = SAE.from_pretrained(
#             sae_id,
#             device=device,
#         )
#         print(f"⚠️ Loaded SAE (old API with sae_id={sae_id})")
#         return sae, getattr(sae, "cfg", None)

#     except Exception as e:
#         raise RuntimeError(
#             f"❌ Failed to load SAE. Your sae-lens version is incompatible.\n"
#             f"Try reinstalling:\n"
#             f"uv pip install -U sae-lens\n\n"
#             f"Original error: {e}"
#         )



def load_sae(layer: int, release: str, device: str):
    print("⚠️ Using FAKE SAE (fallback mode)")
    return None, None