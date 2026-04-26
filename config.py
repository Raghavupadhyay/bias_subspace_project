MODEL_NAME = "gpt2"
LAYER = LAYERS_TO_TEST = [6, 8, 10, 11]
DEVICE = "cpu"  # change to "cpu" if needed

SAE_HIDDEN = 512
LR = 1e-3
EPOCHS = 5

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]