import torch
from typing import Callable, Dict

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


ACTIVATION_FNS: Final[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
    "celu": torch.nn.functional.celu,
    "elu": torch.nn.functional.elu,
    "gelu": torch.nn.functional.gelu,
    "hardswish": torch.nn.functional.hardswish,
    "hardtanh": torch.nn.functional.hardtanh,
    "identity": identity,
    "leaky_relu": torch.nn.functional.leaky_relu,
    "logsigmoid": torch.nn.functional.logsigmoid,
    "log_sigmoid": torch.nn.functional.logsigmoid,
    "none": identity,
    "relu": torch.relu,
    "relu6": torch.nn.functional.relu6,
    "rrelu": torch.nn.functional.rrelu,
    "sigmoid": torch.sigmoid,
    "selu": torch.nn.functional.selu,
    "silu": torch.nn.functional.silu,
    "softplus": torch.nn.functional.softplus,
    "swish": torch.nn.functional.silu,
    "tanh": torch.tanh,
}


def get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Get an activation function by name.
    :param activation: the name of the activation function.

    :param activation: a case-insensitive name of the activation function.
    :returns: an activation function
    """
    activation = activation.lower()
    if activation not in ACTIVATION_FNS:
        raise RuntimeError(
            "Supported activations are `{}`, not {}".format(ACTIVATION_FNS.keys(), activation)
        )

    return ACTIVATION_FNS[activation]
