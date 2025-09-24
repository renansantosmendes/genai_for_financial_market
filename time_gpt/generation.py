import torch
import numpy as np
from config import DEVICE, SEQ_LEN, MIN_SIGMA

def generate_series_stochastic(model, init_series_scaled, future_steps=200, temp=1.0):
    """
    Gera uma série temporal estocástica a partir de uma série inicial.
    """
    model.eval()
    generated = list(init_series_scaled)
    with torch.no_grad():
        for _ in range(future_steps):
            inp = torch.tensor(generated[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, sigma = model(inp)
            mu_t = mu[0, -1].item()
            sigma_t = sigma[0, -1].item()
            sigma_t_adj = max(MIN_SIGMA, sigma_t * temp)
            next_val = np.random.normal(loc=mu_t, scale=sigma_t_adj)
            generated.append(float(next_val))
    return np.array(generated)
