import numpy as np
import random
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for most common sources of randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
