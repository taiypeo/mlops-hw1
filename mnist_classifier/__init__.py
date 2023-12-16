import random

import numpy as np
import torch

from .network import ConvNet  # noqa: F401


def setup_random_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
