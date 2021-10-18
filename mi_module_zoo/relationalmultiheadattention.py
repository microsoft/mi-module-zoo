import math
import torch
from torch import nn
from typing import Optional


class RelationalMultiheadAttention(nn.Module):
    """
    A relational multihead implementation supporting two variations of using additional
    relationship (sparse) information between input elements:

    * Sparse relations (edges):

      * If edges are present and ``edge_attention_bias_is_scalar=False``,``
        and ``use_edge_value_biases=True`` is set, this implements
        Eqs. (3) and (4) of
        Shaw, Peter, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with relative position representations."
        In ACL 2018.  https://www.aclweb.org/anthology/N18-2074/

        and
        Eq. (2) of
        Wang, Bailin, et al. "RAT-SQL: Relation-aware schema encoding and linking for text-to-SQL parsers."
        In ICML 2020.  https://arxiv.org/pdf/1911.04942.pdf

      * If edges are present and ``edge_attention_bias_is_scalar=True``,
        and ``use_edge_value_biases=False`` is set, this implements Sect. 3.1 of
        Hellendoorn, Vincent J., et al. "Global relational modules of source code."
        In ICLR 2020. https://openreview.net/pdf?id=B1lnbRNtwr

    * Dense relations, when all input elements have a relationship information
      to all other elements in the input. This can be encoded in one or both of the
      following two ways:

      * Passing a dense ``dense_relations_kq`` of shape ``[batch_size, query_len, key_len, num_heads]``
        in ``forward()`` for every pair of query-key.
      * Passing a dense ``dense_relations_kv`` of shape ``[batch_size, query_len, key_len, num_heads, value_dimension]``
        in ``forward()`` for every pair of query-key.


    * If no edges are present and no dense relations are passed then
      this acts as a standard multihead attention layer.

    Args:
        num_heads: the number of attention heads.
        num_edge_types: the number of discrete edge types.
        key_query_dimension: the dimensionality of keys and queries (per head).
        value_dimension: the dimension of the values (per head).
        output_dimension: the output dimension (after the feedforward).
        dropout_rate: the rate of dropout in :math:`[0, 1)`.
        use_edge_value_biases: should the edges (relations) use value biases?
        edge_attention_bias_is_scalar: Should edge_attention_biases be a scalar or
            of size ``key_query_dimension``?
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_edge_types: int,
        key_query_dimension: int,
        value_dimension: int,
        output_dimension: int,
        dropout_rate: float,
        use_edge_value_biases: bool = False,
        edge_attention_bias_is_scalar: bool = False,
    ):
        super().__init__()
        assert num_heads > 0
        assert 0 <= dropout_rate < 1

        self._num_heads = num_heads
        self._value_dim = value_dimension
        self._key_query_dim = key_query_dimension

        self._dropout_layer = nn.Dropout(p=dropout_rate)
        self._scaling = key_query_dimension ** -0.5
        self._out_proj = nn.Linear(value_dimension * num_heads, output_dimension, bias=False)

        self._use_edge_value_biases = use_edge_value_biases
        self._edge_attention_bias_is_scalar = edge_attention_bias_is_scalar

        if self._edge_attention_bias_is_scalar:
            edge_attention_bias_dim = num_heads
        else:
            edge_attention_bias_dim = num_heads * key_query_dimension
        self._edge_attention_biases = nn.Embedding(
            num_embeddings=num_edge_types, embedding_dim=edge_attention_bias_dim
        )

        if self._use_edge_value_biases:
            self._edge_value_biases = nn.Embedding(
                num_embeddings=num_edge_types, embedding_dim=num_heads * value_dimension
            )

    @property
    def num_edge_types(self) -> int:
        return self._edge_attention_biases.num_embeddings

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def value_dim(self) -> int:
        return self._value_dim

    @property
    def key_query_dim(self) -> int:
        return self._key_query_dim

    def forward(
        self,
        *,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        masked_elements: Optional[torch.Tensor],
        edges: torch.Tensor,
        edge_types: torch.Tensor,
        dense_relations_kq: Optional[torch.Tensor] = None,
        dense_relations_kv: Optional[torch.Tensor] = None,
    ):
        """
        :param queries: ``[batch_size, query_len, D]``
        :param keys: ``[batch_size, key_len, D]``
        :param values: ``[batch_size, key_len, H]``
        :param masked_elements: bool tensor of shape ``[batch_size, key_len]`` or ``[batch_size, query_len, key_len]``
            ``True`` values are those that should be masked (no attention paid). ``None`` keeps everything unmasked.
        :param edges: ``[num_edges, 3]`` each row has the form ``(batch_idx, source_idx, target_idx)``
        :param edge_types: ``[num_edges]`` of integers from ``0..num_edge_types``
        :param dense_relations_kq: Optional ``[batch_size, query_len, key_len, num_heads]``
        :param dense_relations_kv: Optional ``[batch_size, query_len, key_len, num_heads, value_dimension]``
        :return: ``[batch_size, seq_size, H]``
        """
        edge_sample_ids = edges[:, 0]
        edge_sources = edges[:, 1]
        edge_targets = edges[:, 2]

        # Standard dot-attention: Here, we compute
        #    e_bijk = (in_bi * W_Q^k) * (in_bj * W_K^k)^T
        # i.e., the inner product of the query-projection of token in_bi and key-projection of token in_bj,
        # where b is the ID of the sample in the batch, i, j are token IDs, and k is the ID of a head.
        raw_attention_scores = torch.einsum(
            "bkhd,bqhd->bqkh", keys, queries
        )  # [B, query_len, memory_len, num_heads]

        if edge_sample_ids.shape[0] > 0:
            attention_scores = self._add_edge_attention_scores(
                edge_sample_ids,
                edge_sources,
                edge_targets,
                edge_types,
                keys,
                queries,
                raw_attention_scores,
            )
        else:
            attention_scores = raw_attention_scores

        if dense_relations_kq is not None:
            attention_scores = attention_scores + dense_relations_kq

        attention_scores = attention_scores.transpose(2, 3)  # [B, query_len, num_heads, key_len]

        if masked_elements is not None:
            if masked_elements.dim() == 2:
                attention_scores.masked_fill_(masked_elements.unsqueeze(1).unsqueeze(1), -math.inf)
            else:
                attention_scores.masked_fill_(masked_elements.unsqueeze(2), -math.inf)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self._dropout_layer(attention_probs)

        multiheaded_weighted_value_sum = torch.einsum(
            "blhq,bqhd->blhd", attention_probs, values
        )  # [B, query_len, num_heads, value_dim]

        if self._use_edge_value_biases:
            multiheaded_weighted_value_sum = self._add_edge_value_biases(
                edge_sample_ids,
                edge_sources,
                edge_targets,
                edge_types,
                attention_probs,
                multiheaded_weighted_value_sum,
            )
        if dense_relations_kv is not None:
            value_biases = torch.einsum("bqhk,bqkhd->bqhd", attention_probs, dense_relations_kv)
            multiheaded_weighted_value_sum = multiheaded_weighted_value_sum + value_biases

        attention_output = multiheaded_weighted_value_sum.reshape(
            (
                multiheaded_weighted_value_sum.shape[0],
                multiheaded_weighted_value_sum.shape[1],
                -1,
            )
        )  # [B, query_len, num_heads * value_dim]
        return self._out_proj(attention_output)  # [B, query_len, out_dim]

    def _add_edge_attention_scores(
        self,
        edge_sample_ids,
        edge_sources,
        edge_targets,
        edge_types,
        keys,
        queries,
        raw_attention_scores,
    ):
        # We compute (sparse, per existing edge) additional bias scores e'_bijk:
        edge_bias_scores = self._compute_edge_bias_scores(
            edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries
        )

        # We add the e'_bijk (where present) to e_bijk. This should be a simple +=, but
        # that doesn't accumulate if we have several entries to add to e_bij. Hence we use
        # index_put_, which in turn requires a contiguous Tensor memory layout, and so we need
        # to establish that first:
        attention_scores = raw_attention_scores.contiguous()
        edge_sample_indices = edge_sample_ids
        edge_query_indices = edge_sources
        edge_key_indices = edge_targets
        attention_scores.index_put_(
            indices=(edge_sample_indices, edge_query_indices, edge_key_indices),
            values=edge_bias_scores,
            accumulate=True,
        )

        return attention_scores

    def _compute_edge_bias_scores(
        self, edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries
    ):
        # We will compute additional e'_bihj which will be added onto the standard attention scores:
        attention_biases = self._edge_attention_biases(edge_types)

        if self._edge_attention_bias_is_scalar:
            # Compute e'_bijk = \sum_d (bias_bijk * (in_bj * W_K^k))_d
            # This is the GREAT model. Note two things:
            #  (1) This is defined on the _key_ representation, not the _query_ repr.
            #  (2) Because bias_bijk is a scalar, this is essentially just scaling
            #      (in_bj * W_K^k) and then summing.
            edge_bias_scores = torch.einsum(
                "eh,ehd->eh",
                attention_biases,  # [num_edges, num_heads]
                keys[edge_sample_ids, edge_targets],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
        else:
            # Compute e'_bijk = (in_bj * W_Q^k) * bias_bijk^T
            # This is the Relative Position Representations / RAT-SQL variant. Note that this
            # is defined using the query representation, not the key repr.
            edge_bias_scores = torch.einsum(
                "ehd,ehd->eh",
                attention_biases.reshape((-1, self._num_heads, self._key_query_dim)),
                # [num_edges, num_heads, key_dim]
                keys[edge_sample_ids, edge_targets],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
        return edge_bias_scores

    def _add_edge_value_biases(
        self,
        edge_sample_ids,
        edge_sources,
        edge_targets,
        edge_types,
        attention_probs,
        multiheaded_weighted_value_sum,
    ):
        edge_sample_indices = edge_sample_ids
        edge_query_indices = edge_sources

        value_biases_shape = (
            edge_sample_ids.shape[0],
            self._num_heads,
            self._value_dim,
        )
        value_bias_per_edge = attention_probs[
            edge_sample_ids, edge_sources, :, edge_targets
        ].unsqueeze(-1) * self._edge_value_biases(edge_types).reshape(
            value_biases_shape
        )  # [num_edges, num_heads, value_dim]

        biased_weighted_value_sum = multiheaded_weighted_value_sum.contiguous()
        biased_weighted_value_sum.index_put_(
            indices=(edge_sample_indices, edge_query_indices),
            values=value_bias_per_edge,
            accumulate=True,
        )
        return biased_weighted_value_sum
