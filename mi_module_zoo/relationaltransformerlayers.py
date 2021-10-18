import torch
from torch import nn
from typing import Callable, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from mi_module_zoo.relationalmultiheadattention import RelationalMultiheadAttention
from mi_module_zoo.utils.activation import get_activation_fn


class RelationalTransformerEncoderLayer(nn.Module):
    """
    A relational transformer encoder layer. That supports both discrete/sparse edge types
    and dense (all-to-all) relations, different ReZero modes, and different normalization
    modes.

    Args:
        d_model: the dimensionality of the inputs/ouputs of the transformer layer.
        key_query_dimension: the dimensionality of key/queries in the multihead attention.
        value_dimension: the dimensionality of the multihead attention values,
        num_heads: the number of attention heads,
        num_edge_types: the number of discrete edge types. If ``0``, no discrete edge types
            are to be used.
        add_reverse_edges: if ``num_edge_types>0`` should reverse edge types be introduced?
        dim_feedforward: the dimensionality of the feedforward hidden layer in the transformer layer.
        dropout_rate: the dropout rate in :math:`[0, 1)`,
        activation: the activation function to be used in the feedforward layer. Defaults to ReLU.
        use_edge_value_biases: should the discrete edges (relations) use value biases?
        edge_attention_bias_is_scalar: should ``edge_attention_biases`` be a scalar or
          of size ``key_query_dimension``?
        rezero_mode: Three different modes are supported

          * ``"off"``: No ReZero use.
          * ``"scalar"``: Sublayers (attention / fully connected) are scaled by a single scalar, i.e.,
            ``alpha`` is a scalar in the following: ::

                      x' = x + alpha * SelfAtt(x)
                      x'' = x' + alpha * Boom(x')
                      return x''

            See https://arxiv.org/pdf/2003.04887.pdf.
          * ``"vector"``: Sublayers (attention / fully connected) are scaled by one value per dim, i.e.,
            ``alpha`` is a vector in the following: ::

                        x' = x + alpha * SelfAtt(x)
                        x'' = x' + alpha * Boom(x')
                        return x''

            See https://arxiv.org/pdf/2103.17239.pdf.

        normalisation_mode: Three different modes are supported:

          * ``"off"``: use no layer norm at all. Likely to diverge without using ReZero as well.
          * ``"prenorm"``: Normalise values before each sublayer (attention / fully connected): ::

                    x' = x + SelfAtt(LN(x))
                    x'' = x' + Boom(LN(x'))
                    return x''

          * ``"postnorm"``: Normalise values after each sublayer: ::

                    x' = LN(x + SelfAtt(x))
                    x'' = LN(x' + Boom(x))
                    return x''
    """

    def __init__(
        self,
        d_model: int,
        key_query_dimension: int,
        value_dimension: int,
        num_heads: int,
        num_edge_types: int,
        add_reverse_edges: bool = True,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_edge_value_biases: bool = False,
        edge_attention_bias_is_scalar: bool = False,
        rezero_mode: Literal["off", "scalar", "vector", "scalar-tied"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    ):
        super(RelationalTransformerEncoderLayer, self).__init__()
        assert 0 <= dropout_rate < 1

        self.self_attn = RelationalMultiheadAttention(
            num_heads=num_heads,
            output_dimension=d_model,
            dropout_rate=dropout_rate,
            num_edge_types=2 * num_edge_types if add_reverse_edges else num_edge_types,
            key_query_dimension=key_query_dimension,
            value_dimension=value_dimension,
            use_edge_value_biases=use_edge_value_biases,
            edge_attention_bias_is_scalar=edge_attention_bias_is_scalar,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self._selfatt_head_transforms = nn.Linear(
            in_features=d_model,
            out_features=num_heads * (2 * key_query_dimension + value_dimension),
            bias=False,
        )

        self._normalisation_mode = normalisation_mode
        if normalisation_mode in ("prenorm", "postnorm"):
            self.norm1: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
            self.norm2: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
        elif normalisation_mode == "off":
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"Unrecognized normalization mode `{normalisation_mode}`.")

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.activation = get_activation_fn(activation)

        self._rezero_mode = rezero_mode
        if rezero_mode == "off":
            self._alpha1: Union[float, torch.Tensor] = 1.0
            self._alpha2: Union[float, torch.Tensor] = 1.0
        elif rezero_mode == "scalar":
            self._alpha1 = nn.Parameter(torch.tensor(0.0))
            self._alpha2 = nn.Parameter(torch.tensor(0.0))
        elif rezero_mode == "scalar-tied":
            # The original ReZero setting: https://github.com/majumderb/rezero/blob/e2c94a825c5564217e8cf4d75a28d59cab1d7029/rezero/transformer/rztx.py#L47
            self._alpha1 = nn.Parameter(torch.tensor(0.0))
            self._alpha2 = self._alpha1
        elif rezero_mode == "vector":
            self._alpha1 = nn.Parameter(torch.zeros(size=(d_model,)))
            self._alpha2 = nn.Parameter(torch.zeros(size=(d_model,)))
        else:
            raise ValueError(f"Unrecognized rezero mode `{rezero_mode}`.")

        self._num_edge_types = num_edge_types
        self._add_reverse_edges = add_reverse_edges

    def _compute_qkv(
        self, input_seq_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys_queries_values = self._selfatt_head_transforms(input_seq_states).reshape(
            input_seq_states.shape[0],
            input_seq_states.shape[1],
            self.self_attn._num_heads,
            -1,
        )
        queries, keys, values = torch.split(
            keys_queries_values,
            split_size_or_sections=[
                self.self_attn._key_query_dim,
                self.self_attn._key_query_dim,
                self.self_attn._value_dim,
            ],
            dim=-1,
        )  # [B, query_len, num_heads, key_dim], [B, memory_len, num_heads, key_dim], [B, memory_len, num_heads, value_dim]

        return queries, keys, values

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        edges: torch.Tensor,
        edge_types: torch.Tensor,
        dense_relations_kq: Optional[torch.Tensor] = None,
        dense_relations_kv: Optional[torch.Tensor] = None,
        post_self_att_hook: Optional[
            Callable[[torch.Tensor, Union[float, torch.Tensor]], torch.Tensor]
        ] = None,
    ):
        """
        :param src: A ``[batch_size, seq_len, D]`` tensor.
        :param src_mask: A ``[batch_size, seq_len]`` or ``[batch_size, seq_len (query), seq_len (key)]`` bool tensor.
            ``True`` values are those that should be masked (no attention paid).
        :param edges: ``[num_edges, 3]`` each row has the form ``(batch_idx, source_idx, target_idx)``
          or an empty tensor of shape ``(0, 3)`` of sparse edges are unused.
        :param edge_types: ``[num_edges]`` of integers from ``0..num_edge_types-1`` or an empty tensor
          if sparse edges are unused.
        :param dense_relations_kq: Optional ``[batch_size, seq_len, seq_len, num_heads]``.
        :param dense_relations_kv: Optional ``[batch_size, seq_len, seq_len, num_heads, value_dimension]``
        :return:  ``[batch_size, seq_len, D]``
        """
        # --- Sublayer 1: Self-Attention:
        attn_input = src
        if self._normalisation_mode == "prenorm":
            attn_input = self.norm1(src)

        if self._add_reverse_edges:
            # Create reverse edges
            edge_sample_ids = edges[:, 0].repeat(2)
            edge_sources = torch.cat([edges[:, 1], edges[:, 2]])
            edge_targets = torch.cat([edges[:, 2], edges[:, 1]])
            edges = torch.stack((edge_sample_ids, edge_sources, edge_targets), dim=-1)
            edge_types = torch.cat([edge_types, edge_types + self._num_edge_types])

        queries, keys, values = self._compute_qkv(attn_input)
        src2 = self.self_attn(
            queries=queries,
            keys=keys,
            values=values,
            masked_elements=src_mask,
            edges=edges,
            edge_types=edge_types,
            dense_relations_kq=dense_relations_kq,
            dense_relations_kv=dense_relations_kv,
        )
        src2 = self._alpha1 * src2
        src = src + self.dropout1(src2)

        if post_self_att_hook is not None:
            src = post_self_att_hook(src, self._alpha1)

        if self._normalisation_mode == "postnorm":
            src = self.norm1(src)

        fc_input = src
        if self._normalisation_mode == "prenorm":
            fc_input = self.norm2(fc_input)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(fc_input))))
        src2 = self._alpha2 * src2
        src = src + self.dropout2(src2)
        if self._normalisation_mode == "postnorm":
            src = self.norm1(src)
        return src


class RelationalTransformerDecoderLayer(nn.Module):
    """
    A relational transformer decoder layer. See the :class:`.RelationalTransformerEncoderLayer`
    for more information.
    """

    def __init__(
        self,
        d_model: int,
        key_query_dimension: int,
        value_dimension: int,
        num_heads: int,
        num_self_edge_types: int,
        num_edge_types_to_encoder: int,
        add_reverse_edges: bool = True,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        use_edge_value_biases: bool = False,
        edge_attention_bias_is_scalar: bool = False,
        rezero_mode: Literal["off", "scalar", "vector", "scalar-tied"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    ):
        super().__init__()
        self.decoder = RelationalTransformerEncoderLayer(
            d_model=d_model,
            key_query_dimension=key_query_dimension,
            value_dimension=value_dimension,
            num_heads=num_heads,
            num_edge_types=2 * num_self_edge_types if add_reverse_edges else num_self_edge_types,
            add_reverse_edges=add_reverse_edges,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
            activation=activation,
            use_edge_value_biases=use_edge_value_biases,
            edge_attention_bias_is_scalar=edge_attention_bias_is_scalar,
            rezero_mode=rezero_mode,
            normalisation_mode=normalisation_mode,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self._key_query_dim = key_query_dimension
        self._value_dim = value_dimension
        self._multi_head_att_transforms = nn.Linear(
            in_features=d_model,
            out_features=num_heads * (key_query_dimension + value_dimension),
            bias=False,
        )
        self._query_transforms = nn.Linear(
            in_features=d_model, out_features=num_heads * key_query_dimension, bias=False
        )
        self.multihead_attn = RelationalMultiheadAttention(
            num_heads=num_heads,
            output_dimension=d_model,
            dropout_rate=dropout_rate,
            num_edge_types=2 * num_edge_types_to_encoder
            if add_reverse_edges
            else num_edge_types_to_encoder,
            key_query_dimension=key_query_dimension,
            value_dimension=value_dimension,
            use_edge_value_biases=use_edge_value_biases,
            edge_attention_bias_is_scalar=edge_attention_bias_is_scalar,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        self_edges: torch.Tensor,
        self_edge_types: torch.Tensor,
        encoder_edges: torch.Tensor,
        encoder_edge_types: torch.Tensor,
        dense_self_relations_kq: Optional[torch.Tensor] = None,
        dense_self_relations_kv: Optional[torch.Tensor] = None,
        dense_encoder_relations_kq: Optional[torch.Tensor] = None,
        dense_encoder_relations_kv: Optional[torch.Tensor] = None,
    ):
        """
        :param tgt: A ``[batch_size, seq_len, D]`` tensor.
        :param memory: A ``[batch_size, mem_len, D]`` tensor.
        :param tgt_mask: A ``[batch_size, seq_len]`` or ``[batch_size, seq_len, seq_len]`` bool tensor.
            ``True`` values are those that should be masked (no attention paid).
            For "causal" attention, ``tgt_mask`` should be 3D and ``tgt_mask[:, i, j] = i > j``,
            i.e. the upper-triangular elements should be ``True``.
        :param memory_mask: A ``[batch_size, mem_len]`` bool tensor.  ``True`` values are those
            that should be masked (no attention paid).
        :param self_edges: ``[num_self_edges, 3]`` each row has the form ``(batch_idx, source_idx, target_idx)``
          or an empty tensor of shape ``(0, 3)`` of sparse edges are unused..
        :param self_edge_types: ``[num_self_edges]`` of integers from ``0..num_self_edges-1``.
        :param encoder_edges: ``[num_enc_edges, 3]`` each row has the form ``(batch_idx, source_idx, target_idx)``
          or an empty tensor of shape ``(0, 3)`` of sparse edges are unused.
          Note: ``target_idx`` refers to elements in the memory.

        :param encoder_edge_types: ``[num_enc_edges]`` of integers from ``0..num_enc_edges-1``
        :param dense_self_relations_kq: Optional ``[batch_size, seq_len, seq_len, num_heads]`` for the
          relationships within the decoder.
        :param dense_self_relations_kv: Optional ``[batch_size, seq_len, seq_len, num_heads, value_dimension]``
          relationships within the decoder.
        :param dense_encoder_relations_kq: Optional ``[batch_size, seq_len, mem_len, num_heads]``
          relationships between the encoded inputs and the decoder.
        :param dense_encoder_relations_kv: Optional ``[batch_size, seq_len, mem_len, num_heads, value_dimension]``
          relationships between the encoded inputs and the decoder.

        :return:  ``[batch_size, seq_len, H]``
        """

        def callback(src: torch.Tensor, rezero_alpha: Union[float, torch.Tensor]) -> torch.Tensor:
            kv = self._multi_head_att_transforms(memory).reshape(
                memory.shape[0], memory.shape[1], self.multihead_attn._num_heads, -1
            )
            keys, values = (
                kv[:, :, :, : self._key_query_dim],
                kv[:, :, :, self._key_query_dim :],
            )
            queries = self._query_transforms(src).reshape(
                src.shape[0], src.shape[1], self.multihead_attn.num_heads, -1
            )

            src2 = self.multihead_attn(
                queries=queries,
                keys=keys,
                values=values,
                masked_elements=memory_mask,
                edges=encoder_edges,
                edge_types=encoder_edge_types,
                dense_relations_kq=dense_encoder_relations_kq,
                dense_relations_kv=dense_encoder_relations_kv,
            )

            return src + self.dropout(rezero_alpha * src2)

        return self.decoder(
            tgt,
            tgt_mask,
            edges=self_edges,
            edge_types=self_edge_types,
            dense_relations_kq=dense_self_relations_kq,
            dense_relations_kv=dense_self_relations_kv,
            post_self_att_hook=callback,
        )
