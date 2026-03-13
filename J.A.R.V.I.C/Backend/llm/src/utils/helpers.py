import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(prefer_gpu=True):
    """Return torch device."""
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"