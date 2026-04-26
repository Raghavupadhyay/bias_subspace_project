
import torch

import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        f = torch.relu(self.encoder(x))
        recon = self.decoder(f)
        return f, recon