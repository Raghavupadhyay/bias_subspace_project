
from transformer_lens import HookedTransformer

def load_model():
    model = HookedTransformer.from_pretrained("gpt2")
    return model