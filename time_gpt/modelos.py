import torch
import torch.nn as nn
from config import MIN_SIGMA

class ProbabilisticTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(d_model, 1)
        self.fc_sigma = nn.Linear(d_model, 1)
        # Inicialização pequena para sigma
        nn.init.constant_(self.fc_sigma.bias, -5.0)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(-1)                 # (batch, seq_len, 1)
        h = self.input_proj(x)              # (batch, seq_len, d_model)
        h = self.transformer(h)             # (batch, seq_len, d_model)
        mu = self.fc_mu(h).squeeze(-1)      # (batch, seq_len)
        # Para sigma usar softplus (positivo e estável)
        sigma = torch.nn.functional.softplus(self.fc_sigma(h)).squeeze(-1)
        sigma = torch.clamp(sigma, min=MIN_SIGMA)
        return mu, sigma
