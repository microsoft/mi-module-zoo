import torch
import unittest
from itertools import product
from typing import Iterator, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from mi_module_zoo.relationaltransformerlayers import (
    RelationalTransformerDecoderLayer,
    RelationalTransformerEncoderLayer,
)
from mi_module_zoo.utils.testing import TensorSpec, generate_tensors, is_finite_tensor


class RelationalTransformerEncoderSettings(NamedTuple):
    use_dense_relations: bool
    use_sparse_edges: bool
    use_reverse_edges: bool
    use_mask: bool
    rezero_mode: Literal["off", "scalar", "vector", "scalar-tied"]
    normalisation_mode: Literal["off", "prenorm", "postnorm"]


class RelationalTransformerDecoderSettings(NamedTuple):
    use_dense_relations_enc: bool
    use_dense_relations_self: bool
    use_sparse_edges_enc: bool
    use_sparse_edges_self: bool
    use_reverse_edges: bool
    use_mask: bool
    rezero_mode: Literal["off", "scalar", "vector", "scalar-tied"]
    normalisation_mode: Literal["off", "prenorm", "postnorm"]


class TestRelationalTransformers(unittest.TestCase):
    def __relational_transformer_encoder_settings(
        self,
    ) -> Iterator[RelationalTransformerEncoderSettings]:
        """Yield all valid configurations of relational transformers."""
        for use_mask, use_dense_relations, use_sparse_edges in product((False, True), repeat=3):
            for rezero_mode in ("off", "scalar", "vector", "scalar-tied"):
                for norm_mode in ("off", "prenorm", "postnorm"):
                    if use_sparse_edges:
                        for use_reverse_edges in (False, True):
                            yield RelationalTransformerEncoderSettings(
                                use_dense_relations=use_dense_relations,
                                use_sparse_edges=True,
                                use_reverse_edges=use_reverse_edges,
                                use_mask=use_mask,
                                rezero_mode=rezero_mode,
                                normalisation_mode=norm_mode,
                            )
                    else:
                        yield RelationalTransformerEncoderSettings(
                            use_dense_relations=use_dense_relations,
                            use_sparse_edges=False,
                            use_reverse_edges=False,
                            use_mask=use_mask,
                            rezero_mode=rezero_mode,
                            normalisation_mode=norm_mode,
                        )

    def __relational_transformer_decoder_settings(
        self,
    ) -> Iterator[RelationalTransformerDecoderSettings]:
        """Yield all valid configurations of relational transformers."""
        for (
            use_mask,
            use_dense_relations_enc,
            use_dense_relations_self,
            use_sparse_edges_enc,
            use_sparse_edges_self,
        ) in product((False, True), repeat=5):
            for rezero_mode in ("off", "scalar", "vector", "scalar-tied"):
                for norm_mode in ("off", "prenorm", "postnorm"):
                    if use_sparse_edges_self:
                        for use_reverse_edges in (False, True):
                            yield RelationalTransformerDecoderSettings(
                                use_dense_relations_enc=use_dense_relations_enc,
                                use_dense_relations_self=use_dense_relations_self,
                                use_sparse_edges_self=True,
                                use_sparse_edges_enc=use_sparse_edges_enc,
                                use_reverse_edges=use_reverse_edges,
                                use_mask=use_mask,
                                rezero_mode=rezero_mode,
                                normalisation_mode=norm_mode,
                            )
                    else:
                        yield RelationalTransformerDecoderSettings(
                            use_dense_relations_enc=use_dense_relations_enc,
                            use_dense_relations_self=use_dense_relations_self,
                            use_sparse_edges_self=False,
                            use_sparse_edges_enc=use_sparse_edges_enc,
                            use_reverse_edges=False,
                            use_mask=use_mask,
                            rezero_mode=rezero_mode,
                            normalisation_mode=norm_mode,
                        )

    def test_relational_encoder(self):
        input_spec = {
            "src": TensorSpec("batch_size", "len", "D"),
            "mask": TensorSpec("batch_size", "len"),
            # Sparse relational edges
            "edges_batch_idx": TensorSpec("num_edges", indexed_dim="batch_size"),
            "edges_from_idx": TensorSpec("num_edges", indexed_dim="len"),
            "edges_to_idx": TensorSpec("num_edges", indexed_dim="len"),
            "edge_types": TensorSpec("num_edges", indexed_dim="num_edge_types"),
            # Dense relation edges
            "dense_relations_kq": TensorSpec("batch_size", "len", "len", "num_heads"),
            "dense_relations_kv": TensorSpec(
                "batch_size", "len", "len", "num_heads", "value_dimension"
            ),
        }
        shapes, tensors = generate_tensors(
            input_spec,
            other_dims={
                "key_query_dimension",
                "value_dimension",
                "num_heads",
                "num_edge_types",
                "dim_feedforward",
            },
        )
        print(shapes)
        sparse_edges = torch.stack(
            [tensors["edges_batch_idx"], tensors["edges_from_idx"], tensors["edges_to_idx"]], dim=-1
        )
        self.assertEqual(sparse_edges.shape, (shapes["num_edges"], 3))

        for rel_transformer_setting in self.__relational_transformer_encoder_settings():
            with self.subTest(**rel_transformer_setting._asdict()):
                print(rel_transformer_setting)
                encoder = RelationalTransformerEncoderLayer(
                    d_model=shapes["D"],
                    key_query_dimension=shapes["key_query_dimension"],
                    value_dimension=shapes["value_dimension"],
                    num_heads=shapes["num_heads"],
                    num_edge_types=shapes["num_edge_types"],
                    add_reverse_edges=rel_transformer_setting.use_reverse_edges,
                    dim_feedforward=shapes["dim_feedforward"],
                    use_edge_value_biases=rel_transformer_setting.use_sparse_edges,
                    rezero_mode=rel_transformer_setting.rezero_mode,
                    normalisation_mode=rel_transformer_setting.normalisation_mode,
                )

                if rel_transformer_setting.use_sparse_edges:
                    edges = sparse_edges
                    edge_types = tensors["edge_types"]
                else:
                    edges = torch.zeros((0, 3), dtype=torch.int64)
                    edge_types = torch.zeros(0, dtype=torch.int64)

                dense_relations_kq = (
                    tensors["dense_relations_kq"]
                    if rel_transformer_setting.use_dense_relations
                    else None
                )
                dense_relations_kv = (
                    tensors["dense_relations_kv"]
                    if rel_transformer_setting.use_dense_relations
                    else None
                )
                mask = tensors["mask"] > 0.5 if rel_transformer_setting.use_mask else None
                if mask is not None:
                    mask[:, 0] = False  # Enforce at least one element per set.

                out = encoder(
                    src=tensors["src"],
                    src_mask=mask,
                    edges=edges,
                    edge_types=edge_types,
                    dense_relations_kq=dense_relations_kq,
                    dense_relations_kv=dense_relations_kv,
                )

                self.assertEqual(out.shape, (shapes["batch_size"], shapes["len"], shapes["D"]))
                self.assertTrue(is_finite_tensor(out))

    def test_relational_decoder(self):
        """
        :param tgt: A [batch_size, seq_len, D] tensor.
        :param memory: A [batch_size, mem_len, D] tensor.
        :param tgt_mask: A [batch_size, seq_len] bool tensor
        :param self_edges: [num_self_edges, 3] each row has the form (batch_idx, source_idx, target_idx)
        :param self_edge_types: [num_self_edges] of integers from 0..num_self_edges
        :param encoder_edges: [num_enc_edges, 3] each row has the form (batch_idx, source_idx, target_idx)
        :param encoder_edge_types: [num_enc_edges] of integers from 0..num_enc_edges
        :param dense_self_relations_kq: [batch_size, seq_len, seq_len, num_heads]
        :param dense_self_relations_kv: [batch_size, seq_len, seq_len, num_heads, value_dimension]
        :param dense_encoder_relations_kq: [batch_size, seq_len, mem_len, num_heads]
        :param dense_encoder_relations_kv: [batch_size, seq_len, mem_len, num_heads, value_dimension]

        :return:  [batch_size, seq_len, H]
        """
        input_spec = {
            "tgt": TensorSpec("batch_size", "seq_len", "D"),
            "memory": TensorSpec("batch_size", "mem_len", "D"),
            "tgt_mask": TensorSpec("batch_size", "seq_len"),
            "tgt_mask_2d": TensorSpec("batch_size", "seq_len", "seq_len"),
            "memory_mask": TensorSpec("batch_size", "mem_len"),
            # Sparse self-relational edges
            "self_edges_batch_idx": TensorSpec("num_edges", indexed_dim="batch_size"),
            "self_edges_from_idx": TensorSpec("num_edges", indexed_dim="seq_len"),
            "self_edges_to_idx": TensorSpec("num_edges", indexed_dim="seq_len"),
            "self_edge_types": TensorSpec("num_edges", indexed_dim="num_self_edge_types"),
            # Sparse relational edges to memories
            "encoder_edges_batch_idx": TensorSpec("num_edges", indexed_dim="batch_size"),
            "encoder_edges_from_idx": TensorSpec("num_edges", indexed_dim="seq_len"),
            "encoder_edges_to_idx": TensorSpec("num_edges", indexed_dim="mem_len"),
            "encoder_edge_types": TensorSpec("num_edges", indexed_dim="num_edge_types_to_encoder"),
            # Dense relation edges
            "dense_self_relations_kq": TensorSpec("batch_size", "seq_len", "seq_len", "num_heads"),
            "dense_self_relations_kv": TensorSpec(
                "batch_size", "seq_len", "seq_len", "num_heads", "value_dimension"
            ),
            "dense_encoder_relations_kq": TensorSpec(
                "batch_size", "seq_len", "mem_len", "num_heads"
            ),
            "dense_encoder_relations_kv": TensorSpec(
                "batch_size", "seq_len", "mem_len", "num_heads", "value_dimension"
            ),
        }

        shapes, tensors = generate_tensors(
            input_spec,
            other_dims={
                "key_query_dimension",
                "value_dimension",
                "num_heads",
                "num_self_edge_types",
                "num_edge_types_to_encoder",
                "dim_feedforward",
            },
        )
        print(shapes)
        sparse_self_edges = torch.stack(
            [
                tensors["self_edges_batch_idx"],
                tensors["self_edges_from_idx"],
                tensors["self_edges_to_idx"],
            ],
            dim=-1,
        )
        self.assertEqual(sparse_self_edges.shape, (shapes["num_edges"], 3))

        sparse_encoder_edges = torch.stack(
            [
                tensors["encoder_edges_batch_idx"],
                tensors["encoder_edges_from_idx"],
                tensors["encoder_edges_to_idx"],
            ],
            dim=-1,
        )
        self.assertEqual(sparse_encoder_edges.shape, (shapes["num_edges"], 3))

        for rel_transformer_setting in self.__relational_transformer_decoder_settings():
            with self.subTest(**rel_transformer_setting._asdict()):
                print(rel_transformer_setting)
                decoder = RelationalTransformerDecoderLayer(
                    d_model=shapes["D"],
                    key_query_dimension=shapes["key_query_dimension"],
                    value_dimension=shapes["value_dimension"],
                    num_heads=shapes["num_heads"],
                    num_self_edge_types=shapes["num_self_edge_types"],
                    num_edge_types_to_encoder=shapes["num_edge_types_to_encoder"],
                    add_reverse_edges=rel_transformer_setting.use_reverse_edges,
                    dim_feedforward=shapes["dim_feedforward"],
                    use_edge_value_biases=rel_transformer_setting.use_sparse_edges_enc,
                    rezero_mode=rel_transformer_setting.rezero_mode,
                    normalisation_mode=rel_transformer_setting.normalisation_mode,
                )

                if rel_transformer_setting.use_sparse_edges_self:
                    self_edges = sparse_self_edges
                    self_edge_types = tensors["self_edge_types"]
                else:
                    self_edges = torch.zeros((0, 3), dtype=torch.int64)
                    self_edge_types = torch.zeros(0, dtype=torch.int64)

                if rel_transformer_setting.use_sparse_edges_enc:
                    encoder_edges = sparse_encoder_edges
                    encoder_edge_types = tensors["encoder_edge_types"]
                else:
                    encoder_edges = torch.zeros((0, 3), dtype=torch.int64)
                    encoder_edge_types = torch.zeros(0, dtype=torch.int64)

                dense_self_relations_kq = (
                    tensors["dense_self_relations_kq"]
                    if rel_transformer_setting.use_dense_relations_self
                    else None
                )
                dense_self_relations_kv = (
                    tensors["dense_self_relations_kv"]
                    if rel_transformer_setting.use_dense_relations_self
                    else None
                )
                dense_encoder_relations_kq = (
                    tensors["dense_encoder_relations_kq"]
                    if rel_transformer_setting.use_dense_relations_enc
                    else None
                )
                dense_encoder_relations_kv = (
                    tensors["dense_encoder_relations_kv"]
                    if rel_transformer_setting.use_dense_relations_enc
                    else None
                )
                tgt_mask = tensors["tgt_mask"] > 0.5 if rel_transformer_setting.use_mask else None
                if tgt_mask is not None:
                    # Enforce at least one element per set.
                    tgt_mask[:, 0] = False

                tgt_mask_2d = (
                    tensors["tgt_mask_2d"] < 0.2 if rel_transformer_setting.use_mask else None
                )
                if tgt_mask_2d is not None:
                    # Enforce attending to self
                    tgt_mask_2d[
                        :, torch.arange(tgt_mask.shape[1]), torch.arange(tgt_mask.shape[1])
                    ] = False

                memory_mask = (
                    tensors["memory_mask"] > 0.5 if rel_transformer_setting.use_mask else None
                )
                if memory_mask is not None:
                    # Enforce at least one element per set.
                    memory_mask[:, 0] = False

                out = decoder(
                    tgt=tensors["tgt"],
                    memory=tensors["memory"],
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    self_edges=self_edges,
                    self_edge_types=self_edge_types,
                    encoder_edges=encoder_edges,
                    encoder_edge_types=encoder_edge_types,
                    dense_self_relations_kq=dense_self_relations_kq,
                    dense_self_relations_kv=dense_self_relations_kv,
                    dense_encoder_relations_kq=dense_encoder_relations_kq,
                    dense_encoder_relations_kv=dense_encoder_relations_kv,
                )

                self.assertEqual(out.shape, (shapes["batch_size"], shapes["seq_len"], shapes["D"]))
                self.assertTrue(is_finite_tensor(out))

                if tgt_mask_2d is not None:
                    # Test 2D mask
                    out = decoder(
                        tgt=tensors["tgt"],
                        memory=tensors["memory"],
                        tgt_mask=tgt_mask_2d,
                        memory_mask=memory_mask,
                        self_edges=self_edges,
                        self_edge_types=self_edge_types,
                        encoder_edges=encoder_edges,
                        encoder_edge_types=encoder_edge_types,
                        dense_self_relations_kq=dense_self_relations_kq,
                        dense_self_relations_kv=dense_self_relations_kv,
                        dense_encoder_relations_kq=dense_encoder_relations_kq,
                        dense_encoder_relations_kv=dense_encoder_relations_kv,
                    )

                    self.assertEqual(
                        out.shape, (shapes["batch_size"], shapes["seq_len"], shapes["D"])
                    )
                    self.assertTrue(is_finite_tensor(out))
