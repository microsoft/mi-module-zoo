import unittest

from mi_module_zoo.utils.activation import ACTIVATION_FNS
from mi_module_zoo.utils.testing import TensorSpec, generate_tensors, is_finite_tensor


class TestActivations(unittest.TestCase):
    def test_activation(self):
        input_spec = {
            "input1": TensorSpec("A", "B"),
            "input2": TensorSpec(
                "C",
            ),
        }
        shapes, tensors = generate_tensors(
            input_spec,
        )

        for activation_name, activation_fn in ACTIVATION_FNS.items():
            with self.subTest(f"Activation {activation_name}."):
                print(activation_name)
                for input in (tensors["input1"], tensors["input2"]):
                    result = activation_fn(input)
                    self.assertEqual(result.shape, input.shape)
                    self.assertTrue(is_finite_tensor(result))
