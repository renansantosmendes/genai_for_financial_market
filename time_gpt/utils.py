import random
import numpy as np
import torch
from config import SEED

def set_seed():
    """Define a semente para reprodutibilidade."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def to_numpy(seq):
    """Converte qualquer objeto em numpy 1D"""
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    elif not isinstance(seq, np.ndarray):
        seq = np.array(seq)

    if seq.ndim == 0:  # escalar
        seq = np.array([seq])
    elif seq.ndim > 1:  # mais de 1 dimensão -> achata
        seq = seq.flatten()
    return seq
