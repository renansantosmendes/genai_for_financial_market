import random
import numpy as np
import torch
from config import SEED

def set_seed():
    """Sets the seed for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def to_numpy(seq):
    """Converts any object to a 1D numpy array."""
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    elif not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    if seq.ndim == 0:
        seq = np.array([seq])
    elif seq.ndim > 1:
        seq = seq.flatten()
    return seq