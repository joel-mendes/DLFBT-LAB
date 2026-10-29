import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=True):
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, and CUDA/MPS as well
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False