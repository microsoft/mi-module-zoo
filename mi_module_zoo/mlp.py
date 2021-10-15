import torch.nn as nn
from typing import List, Sequence


def construct_mlp(
    input_dim: int,
    out_dim: int,
    hidden_layer_dims: Sequence[int],
    activation_layer: nn.Module = nn.ReLU(),
) -> nn.Sequential:
    """
    Construct a multi-linear perceptron (MLP). No non-linearity is *not* applied at the final layer.

    :param input_dim: the input dimension of the MLP.
    :param out_dim: the input dimension of the MLP.
    :param hidden_layer_dims: a list of zero or more integers indicating the dimensions of the
        hidden layers.
    :param activation_layer: the activation layer used between the input and hidden layers.

    :returns: a :class:`nn.Sequential` with the constructed MLP.
    """
    layers: List[nn.Module] = []
    cur_hidden_dim = input_dim
    for hidden_layer_dim in hidden_layer_dims:
        layers.append(nn.Linear(cur_hidden_dim, hidden_layer_dim))
        layers.append(activation_layer)
        cur_hidden_dim = hidden_layer_dim
    layers.append(nn.Linear(cur_hidden_dim, out_dim))
    return nn.Sequential(*layers)
