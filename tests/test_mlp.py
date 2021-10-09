import unittest

from mi_module_zoo.mlp import construct_mlp
from mi_module_zoo.utils.testing import TensorSpec, generate_tensors, is_finite_tensor


class TestMLP(unittest.TestCase):
    def test_mlp(self):
        input_spec = {
            "input": TensorSpec("Batch", "input_dim"),
        }
        shapes, tensors = generate_tensors(
            input_spec, other_dims={"hidden_dim1", "hidden_dim2", "out_dim"}
        )

        hidden_layer_sizes = (shapes["hidden_dim1"], shapes["hidden_dim2"])

        # Test for 0..2 hidden layers
        for num_layers in range(3):
            with self.subTest(f"MLP with {num_layers} hidden layers."):
                mlp = construct_mlp(
                    input_dim=shapes["input_dim"],
                    out_dim=shapes["out_dim"],
                    hidden_layer_dims=hidden_layer_sizes[:num_layers],
                )

                out = mlp(tensors["input"])
                self.assertEqual(out.shape, (shapes["Batch"], shapes["out_dim"]))
                self.assertTrue(is_finite_tensor(out))
