from transformer_lens import HookedTransformer
import torch

def load_model(device):
    model = HookedTransformer.from_pretrained("gpt2")
    model.to(device)
    return model

def get_activation(model, prompt, layer):
    _, cache = model.run_with_cache(prompt)
    act = cache[f"blocks.{layer}.hook_resid_post"]
    return act[:, -1, :]  # last token