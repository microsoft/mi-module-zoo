import unittest
from itertools import product
from typing import Iterator, NamedTuple

from mi_module_zoo.settransformer import (
    ISAB,
    PMA,
    ElementwiseTransformType,
    MultiheadInitType,
    SetTransformer,
)
from mi_module_zoo.utils.testing import TensorSpec, generate_tensors, is_finite_tensor


class TestSetTransformer(unittest.TestCase):
    def test_isab(self):
        input_spec = {
            "x": TensorSpec("batch_size", "set_size", "per_head_embedding_dim", "num_heads"),
            "mask": TensorSpec("batch_size", "set_size"),
        }
        shapes, tensors = generate_tensors(
            input_spec,
            other_dims={
                "num_inducing_points",
            },
        )
        # Constraint in implementation
        embedding_dim = shapes["per_head_embedding_dim"] * shapes["num_heads"]

        mask = tensors["mask"] < 0.5
        mask[:, 0] = False  # Enforce at least one element per set.

        for mha_init, elementwise_transform_type, use_layer_norm in product(
            MultiheadInitType, ElementwiseTransformType, (False, True)
        ):
            with self.subTest(
                f"mha_init={mha_init}, elementwise_transform_type={elementwise_transform_type}, use_layer_norm={use_layer_norm}"
            ):
                isab = ISAB(
                    embedding_dim=embedding_dim,
                    num_heads=shapes["num_heads"],
                    num_inducing_points=shapes["num_inducing_points"],
                    multihead_init_type=mha_init,
                    use_layer_norm=use_layer_norm,
                    elementwise_transform_type=elementwise_transform_type,
                    dropout_rate=0.1,
                )
                x = tensors["x"]
                out = isab(
                    x=x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]),
                    mask=mask,
                )
                self.assertEqual(
                    out.shape, (shapes["batch_size"], shapes["set_size"], embedding_dim)
                )
                self.assertTrue(is_finite_tensor(out))

    def test_pma(self):
        input_spec = {
            "x": TensorSpec("batch_size", "set_size", "per_head_embedding_dim", "num_heads"),
            "mask": TensorSpec("batch_size", "set_size"),
        }
        shapes, tensors = generate_tensors(
            input_spec,
            other_dims={
                "num_seed_vectors",
            },
        )
        # Constraint in implementation
        embedding_dim = shapes["per_head_embedding_dim"] * shapes["num_heads"]
        mask = tensors["mask"] < 0.5
        mask[:, 0] = False  # Enforce at least one element per set.

        for (
            mha_init,
            elementwise_transform_type,
            use_layer_norm,
            use_elementwise_transform_pma,
        ) in product(
            MultiheadInitType,
            ElementwiseTransformType,
            (False, True),
            (False, True),
        ):
            with self.subTest(
                f"mha_init={mha_init}, elementwise_transform_type={elementwise_transform_type}, use_layer_norm={use_layer_norm}, use_elementwise_transform_pma={use_elementwise_transform_pma}"
            ):
                pma = PMA(
                    embedding_dim=embedding_dim,
                    num_heads=shapes["num_heads"],
                    num_seed_vectors=shapes["num_seed_vectors"],
                    multihead_init_type=mha_init,
                    use_layer_norm=use_layer_norm,
                    elementwise_transform_type=elementwise_transform_type,
                    use_elementwise_transform_pma=use_elementwise_transform_pma,
                    dropout_rate=0.1,
                )
                x = tensors["x"]
                out = pma(
                    x=x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]),
                    mask=mask,
                )

                self.assertEqual(
                    out.shape, (shapes["batch_size"], shapes["num_seed_vectors"], embedding_dim)
                )
                self.assertTrue(is_finite_tensor(out))

    class SetTransformerSettings(NamedTuple):
        use_transformer_embedding_dim: bool
        use_isab: bool
        multihead_init_type: MultiheadInitType
        use_layer_norm: bool
        elementwise_transform_type: ElementwiseTransformType
        use_elementwise_transform_pma: bool
        unit_seed_embeddings: bool

    def __set_transformer_settings(self) -> Iterator[SetTransformerSettings]:
        for mha_init, elementwise_transform_type in product(
            MultiheadInitType, ElementwiseTransformType
        ):
            for (
                use_transformer_embedding_dim,
                use_isab,
                use_layer_norm,
                use_elementwise_transform_pma,
                unit_seed_embeddings,
            ) in product((False, True), repeat=5):
                yield TestSetTransformer.SetTransformerSettings(
                    use_transformer_embedding_dim=use_transformer_embedding_dim,
                    use_isab=use_isab,
                    multihead_init_type=mha_init,
                    use_layer_norm=use_layer_norm,
                    elementwise_transform_type=elementwise_transform_type,
                    use_elementwise_transform_pma=use_elementwise_transform_pma,
                    unit_seed_embeddings=unit_seed_embeddings,
                )

    def test_set_transformer(self):
        for setting in self.__set_transformer_settings():
            with self.subTest(**setting._asdict()):
                if setting.use_transformer_embedding_dim:
                    input_spec = {
                        "x": TensorSpec("batch_size", "set_size", "input_embedding_dim"),
                        "mask": TensorSpec("batch_size", "set_size"),
                    }
                    other_dims = {"transformer_embedding_dim_per_head"}
                else:
                    input_spec = {
                        "x": TensorSpec(
                            "batch_size", "set_size", "input_embedding_dim_per_head", "num_heads"
                        ),
                        "mask": TensorSpec("batch_size", "set_size"),
                    }
                    other_dims = set()

                shapes, tensors = generate_tensors(
                    input_spec,
                    other_dims=other_dims
                    | {
                        "num_seed_vectors",
                        "set_embedding_dim",
                        "num_heads",
                        "num_blocks",
                        "num_inducing_points",
                        "set_embedding_dim",
                    },
                )
                print(shapes)
                mask = tensors["mask"] < 0.5
                mask[:, 0] = False  # Enforce at least one element per set.

                # Constraint in implementation
                if setting.use_transformer_embedding_dim:
                    transformer_embedding_dim = (
                        shapes["transformer_embedding_dim_per_head"] * shapes["num_heads"]
                    )
                    input_embedding_dim = shapes["input_embedding_dim"]
                    x = tensors["x"]
                else:
                    input_embedding_dim = (
                        shapes["input_embedding_dim_per_head"] * shapes["num_heads"]
                    )
                    x = tensors["x"]
                    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

                print(setting)
                num_seed_vectors = 1 if setting.unit_seed_embeddings else shapes["num_seed_vectors"]

                set_transformer = SetTransformer(
                    input_embedding_dim=input_embedding_dim,
                    set_embedding_dim=shapes["set_embedding_dim"],
                    transformer_embedding_dim=transformer_embedding_dim
                    if setting.use_transformer_embedding_dim
                    else None,
                    num_heads=shapes["num_heads"],
                    num_blocks=shapes["num_blocks"],
                    num_seed_vectors=num_seed_vectors,
                    use_isab=setting.use_isab,
                    num_inducing_points=shapes["num_inducing_points"] if setting.use_isab else None,
                    multihead_init_type=setting.multihead_init_type,
                    use_layer_norm=setting.use_layer_norm,
                    elementwise_transform_type=setting.elementwise_transform_type,
                    use_elementwise_transform_pma=setting.use_elementwise_transform_pma,
                )

                out = set_transformer(x, mask)
                self.assertEqual(
                    out.shape,
                    (
                        shapes["batch_size"],
                        shapes["set_embedding_dim"],
                    ),
                )
                self.assertTrue(is_finite_tensor(out))
