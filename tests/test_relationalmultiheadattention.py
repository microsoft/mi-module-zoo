import torch
import unittest
from itertools import product
from typing import Iterator, NamedTuple

from mi_module_zoo.relationalmultiheadattention import RelationalMultiheadAttention
from mi_module_zoo.utils.testing import TensorSpec, generate_tensors, is_finite_tensor


class RelationalMhaSettings(NamedTuple):
    use_dense_relations_kq: bool
    use_dense_relations_kv: bool

    use_sparse_edges: bool
    use_edge_value_biases: bool
    edge_attention_bias_is_scalar: bool

    use_mask: bool


class TestMultiheadAttention(unittest.TestCase):
    def __relational_mha_settings(self) -> Iterator[RelationalMhaSettings]:
        """Yield all valid configurations of relational transformers."""
        for use_mask, use_dense_relations_kq, use_dense_relations_kv, use_sparse_edges in product(
            (False, True), repeat=4
        ):
            if use_sparse_edges:
                for use_edge_value_biases, edge_attention_bias_is_scalar in product(
                    (False, True), repeat=2
                ):
                    yield RelationalMhaSettings(
                        use_dense_relations_kq=use_dense_relations_kq,
                        use_dense_relations_kv=use_dense_relations_kv,
                        use_sparse_edges=use_sparse_edges,
                        use_edge_value_biases=use_edge_value_biases,
                        edge_attention_bias_is_scalar=edge_attention_bias_is_scalar,
                        use_mask=use_mask,
                    )
            else:
                yield RelationalMhaSettings(
                    use_dense_relations_kq=use_dense_relations_kq,
                    use_dense_relations_kv=use_dense_relations_kv,
                    use_sparse_edges=use_sparse_edges,
                    use_edge_value_biases=False,
                    edge_attention_bias_is_scalar=False,
                    use_mask=use_mask,
                )

    def test_relational_mfa(self):
        input_spec = {
            "queries": TensorSpec("batch_size", "query_len", "num_heads", "D"),
            "keys": TensorSpec("batch_size", "key_len", "num_heads", "D"),
            "values": TensorSpec("batch_size", "key_len", "num_heads", "H"),
            "mask": TensorSpec("batch_size", "key_len"),
            # Sparse relational edges
            "edges_batch_idx": TensorSpec("num_edges", indexed_dim="batch_size"),
            "edges_from_idx": TensorSpec("num_edges", indexed_dim="query_len"),
            "edges_to_idx": TensorSpec("num_edges", indexed_dim="key_len"),
            "edge_types": TensorSpec("num_edges", indexed_dim="num_edge_types"),
            # Dense relation edges
            "dense_relations_kq": TensorSpec("batch_size", "query_len", "key_len", "num_heads"),
            "dense_relations_kv": TensorSpec(
                "batch_size", "query_len", "key_len", "num_heads", "H"
            ),
        }
        shapes, tensors = generate_tensors(input_spec, other_dims={"num_edge_types", "out_dim"})
        print(shapes)
        sparse_edges = torch.stack(
            [tensors["edges_batch_idx"], tensors["edges_from_idx"], tensors["edges_to_idx"]], dim=-1
        )
        self.assertEqual(sparse_edges.shape, (shapes["num_edges"], 3))

        for rmha_setting in self.__relational_mha_settings():
            with self.subTest(**rmha_setting._asdict()):
                print(rmha_setting)
                mha = RelationalMultiheadAttention(
                    num_heads=shapes["num_heads"],
                    num_edge_types=shapes["num_edge_types"],
                    key_query_dimension=shapes["D"],
                    value_dimension=shapes["H"],
                    output_dimension=shapes["out_dim"],
                    dropout_rate=0,
                    use_edge_value_biases=rmha_setting.use_edge_value_biases,
                    edge_attention_bias_is_scalar=rmha_setting.edge_attention_bias_is_scalar,
                )

                if rmha_setting.use_sparse_edges:
                    edges = sparse_edges
                    edge_types = tensors["edge_types"]
                else:
                    edges = torch.zeros((0, 3), dtype=torch.int64)
                    edge_types = torch.zeros(0, dtype=torch.int64)

                dense_relations_kq = (
                    tensors["dense_relations_kq"] if rmha_setting.use_dense_relations_kq else None
                )
                dense_relations_kv = (
                    tensors["dense_relations_kv"] if rmha_setting.use_dense_relations_kv else None
                )
                mask = tensors["mask"] > 0.5 if rmha_setting.use_mask else None
                if mask is not None:
                    mask[:, 0] = False  # Enforce at least one element per input.

                output = mha(
                    queries=tensors["queries"],
                    keys=tensors["keys"],
                    values=tensors["values"],
                    masked_elements=mask,
                    edges=edges,
                    edge_types=edge_types,
                    dense_relations_kq=dense_relations_kq,
                    dense_relations_kv=dense_relations_kv,
                )
                self.assertEqual(
                    output.shape, (shapes["batch_size"], shapes["query_len"], shapes["out_dim"])
                )
                self.assertTrue(is_finite_tensor(output))
