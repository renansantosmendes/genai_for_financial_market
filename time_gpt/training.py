import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import DEVICE, EPOCHS, LR

def nll_gaussian(y_true, mu, sigma):
    """Calculates the Negative Log-Likelihood for a Gaussian distribution."""
    var = sigma**2
    return torch.mean(0.5 * torch.log(2 * np.pi * var) + ((y_true - mu)**2) / (2 * var))

def train_model(model, loader):
    """Executes the model training loop."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = nll_gaussian(yb, mu, sigma)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - NLL Loss: {epoch_loss:.6f}")
    print("Training finished.")