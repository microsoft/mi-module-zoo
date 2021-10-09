import random
import torch
from typing import Collection, Dict, Optional, Set, Tuple, Union

try:
    from typing import Final
except ImportError:
    from typing_extensions import Final

__all__ = ["TensorSpec", "generate_tensors"]


class TensorSpec:
    def __init__(self, *dims: Union[str, int], indexed_dim: Optional[str] = None):
        """
        A symbolic specification of a tensor.

        :param dims: a variable-sized list of dimensions. Symbolic names (strings) or concrete
            sizes (integers) can be given.
        :param indexed_dim: Optional. If the tensor indexes another dimension, then the symbolic
             name of that dimension. This implies that the tensor is a torch.int64 tensor with
             values in [0, size(indexed_dim)).
        """
        self.dim: Final = dims
        self.indexed_dim: Final = indexed_dim


_PRIMES: Final = [
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
]


def generate_tensors(
    tensor_specs: Dict[str, TensorSpec],
    other_dims: Collection[str] = frozenset(),
    rng: random.Random = random,
) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    """
    Generate random tensors and assign sizes to dimension based on the input specifications,
        to be used for testing.

    All dimensions will be assigned a different concrete shape which is a prime number. Prime
        shape sizes provide some reassurance that each dimension cannot be confused with another
        one, e.g. due to reshaping.

    :param tensor_specs: a dictionary of tensor names with their associated `TensorSpec`s.
    :param other_dims: a set of named dimensions that should be assigned a concrete size.
    :param rng: the random number generator to be used. Default the `random` package.
    :returns: a tuple with two elements
        concrete_dim_sizes: the concrete size of the dimensions in `tensor_specs` and `other_dims`
        generated_tensors: a dictionary with the concrete random tensors defined in `tensor_specs`.
    """

    # Collect all symbolic dims
    dims: Set[str] = set()
    for spec in tensor_specs.values():
        dims.update(s for s in spec.dim if isinstance(s, str))
    dims.update(other_dims)

    # Assign random (first len(dims) prime) sizes to dims
    shape_sizes = dict(zip(dims, rng.sample(_PRIMES[: len(dims)], k=len(dims))))

    # Generate the random test tensors
    out_tensors = {}
    for tensor_name, tensor_spec in tensor_specs.items():
        shape = tuple(shape_sizes[dim] if isinstance(dim, str) else dim for dim in tensor_spec.dim)
        if tensor_spec.indexed_dim is None:
            tensor = torch.randn(shape)
        else:
            assert (
                tensor_spec.indexed_dim in dims
            ), f"Indexed dimension {tensor_spec.indexed_dim} was not specified in input specs."
            tensor = torch.randint(shape_sizes[tensor_spec.indexed_dim], shape)

        out_tensors[tensor_name] = tensor

    return shape_sizes, out_tensors


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    return not torch.any(torch.isnan(tensor) | torch.isinf(tensor))
