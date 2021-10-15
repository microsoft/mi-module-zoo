import math
import torch
from enum import Enum
from torch import nn
from typing import Optional, cast

__all__ = ["MultiheadInitType", "ElementwiseTransformType", "SetTransformer", "ISAB", "PMA"]


class MultiheadInitType(Enum):
    """Initialization type for multihead attetion."""

    XAVIER = 0
    KAIMING = 1


class ElementwiseTransformType(Enum):
    SINGLE = 0
    DOUBLE = 1


def _initialise_multihead(
    multihead: nn.MultiheadAttention, multihead_init_type: MultiheadInitType
) -> None:
    if multihead_init_type == MultiheadInitType.XAVIER:  # AIAYN and nn.MultiheadAttention default
        nn.init.xavier_uniform_(multihead.in_proj_weight)
        nn.init.constant_(multihead.in_proj_bias, 0.0)

        nn.init.kaiming_uniform_(multihead.out_proj.weight, a=math.sqrt(5))
        nn.init.constant_(multihead.out_proj.bias, 0.0)

    elif multihead_init_type == MultiheadInitType.KAIMING:
        # ST Implementation (nn.Linear) default
        nn.init.kaiming_uniform_(multihead.in_proj_weight, a=math.sqrt(5))
        in_proj_fan_in, _ = nn.init._calculate_fan_in_and_fan_out(multihead.in_proj_weight)
        in_proj_bound = 1 / math.sqrt(in_proj_fan_in)
        nn.init.uniform_(multihead.in_proj_bias, -in_proj_bound, in_proj_bound)

        nn.init.kaiming_uniform_(multihead.out_proj.weight, a=math.sqrt(5))
        out_proj_fan_in, _ = nn.init._calculate_fan_in_and_fan_out(multihead.out_proj.weight)
        out_proj_bound = 1 / math.sqrt(out_proj_fan_in)
        nn.init.uniform_(multihead.out_proj.bias, -out_proj_bound, out_proj_bound)
    else:
        raise ValueError(f"Unrecognized init type `{multihead_init_type}`.")


def _create_elementwise_transform(
    embedding_dim: int, elementwise_transform_type: ElementwiseTransformType
) -> nn.Sequential:
    if elementwise_transform_type == ElementwiseTransformType.SINGLE:  # ST Implementation default
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    elif elementwise_transform_type == ElementwiseTransformType.DOUBLE:
        # AIAYN Implementation default
        return nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    else:
        raise ValueError(f"Unrecognized elementwise transform type `{elementwise_transform_type}`.")


class SetTransformer(nn.Module):
    """
    The Set Transformer model https://arxiv.org/abs/1810.00825

    Generates an embedding from a set of features using several blocks of self attention
    and pooling by attention.

    Args:
            input_embedding_dim: Dimension of the input data, the embedded features.
            set_embedding_dim: Dimension of the output data, the set embedding.
            transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
            num_heads: Number of heads in each multi-head attention block.
            num_blocks: Number of SABs in the model.
            num_seed_vectors: Number of seed vectors used in the pooling block (PMA).
            use_isab: Should ISAB blocks be used instead of SAB blocks.
            num_inducing_points: Number of inducing points.
            multihead_init_type: How linear layers in nn.MultiheadAttention are initialised. Valid options are "xavier" and "kaiming".
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used. Valid options are "single" and "double".
            use_elementwise_transform_pma: Whether an elementwise transform (rFF) should be used in the PMA block.

    """

    def __init__(
        self,
        input_embedding_dim: int,
        set_embedding_dim: int,
        transformer_embedding_dim: Optional[int] = None,
        num_heads: int = 1,
        num_blocks: int = 2,
        num_seed_vectors: int = 1,
        use_isab: bool = False,
        num_inducing_points: Optional[int] = None,
        multihead_init_type: MultiheadInitType = MultiheadInitType.XAVIER,
        use_layer_norm: bool = True,
        elementwise_transform_type: ElementwiseTransformType = ElementwiseTransformType.SINGLE,
        use_elementwise_transform_pma: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self._transform_input_dimension = transformer_embedding_dim is not None
        if transformer_embedding_dim is not None:
            transformer_embedding_dim = cast(int, transformer_embedding_dim)
            assert (
                input_embedding_dim != transformer_embedding_dim
            ), "The linear transformation seems unnecessary since input_embedding_dim == transformer_embedding_dim"
            self._input_dimension_transform = nn.Linear(
                input_embedding_dim, transformer_embedding_dim
            )
        else:
            transformer_embedding_dim = input_embedding_dim

        if use_isab:
            if num_inducing_points is None:
                raise ValueError("Number of inducing points must be defined to use ISAB")
            self._sabs = nn.ModuleList(
                [
                    ISAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        num_inducing_points=num_inducing_points,
                        multihead_init_type=multihead_init_type,
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=elementwise_transform_type,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self._sabs = nn.ModuleList(
                [
                    SAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        multihead_init_type=multihead_init_type,
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=elementwise_transform_type,
                        dropout_rate=dropout_rate,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self._pma = PMA(
            embedding_dim=transformer_embedding_dim,
            num_heads=num_heads,
            num_seed_vectors=num_seed_vectors,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
            use_elementwise_transform_pma=use_elementwise_transform_pma,
            dropout_rate=dropout_rate,
        )
        self._output_dimension_transform = nn.Linear(
            transformer_embedding_dim * num_seed_vectors, set_embedding_dim
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Embedded features tensor with shape ``[batch_size, set_size, input_embedding_dim]``.
            mask: Mask tensor with shape ``[batch_size, set_size]``, ``True`` values are masked
        Returns:
            Set embedding tensor with shape ``[batch_size, set_embedding_dim]``.
        """
        batch_size, _, _ = x.shape

        if self._transform_input_dimension:
            x = self._input_dimension_transform(
                x
            )  # Shape (batch_size, set_size, transformer_embedding_dim)
        for sab in self._sabs:
            x = sab(x, mask)  # Shape (batch_size, set_size, transformer_embedding_dim)
        x = self._pma(x, mask)  # Shape (batch_size, num_seed_vectors, transformer_embedding_dim)
        x = x.reshape(
            (batch_size, -1)
        )  # Shape (batch_size, num_seed_vectors * transformer_embedding_dim)
        set_embedding = self._output_dimension_transform(x)  # Shape (batch_size, set_embedding_dim)

        return set_embedding


class MAB(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        multihead_init_type: MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: ElementwiseTransformType,
        dropout_rate: float,
    ):
        """
        Multihead Attention Block of the Set Transformer model.

        :param embedding_dim: Dimension of the input data.
        :param num_heads: Number of heads.
        :param multihead_init_type: How linear layers in nn.MultiheadAttention are initialised.
        :param use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        :param elementwise_transform_type: Elementwise transform (rFF) type used.
        :param dropout_rate: the percent of elements to dropout.
        """
        super().__init__()
        self._multihead = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        _initialise_multihead(self._multihead, multihead_init_type)

        self._use_layer_norm = use_layer_norm
        if self._use_layer_norm:
            self._layer_norm_1 = nn.LayerNorm(embedding_dim)
            self._layer_norm_2 = nn.LayerNorm(embedding_dim)

        self._elementwise_transform = _create_elementwise_transform(
            embedding_dim, elementwise_transform_type
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        A multi-head attention block with keys==values. See Equation 6 of the Set Transformers paper.

        :param query: Query tensor with shape [batch_size, query_set_size, embedding_dim]
        :param key: Input tensor with shape [batch_size, key_set_size, embedding_dim] to be used as key and value.
        :param key_mask: Boolean mask tensor with shape [batch_size, key_set_size]. True values are masked.
                If None, nothing is masked.
        Returns:
            output: Tensor with shape [batch_size, query_set_size, embedding_dim].
        """
        x = (
            query
            + self._multihead(
                query=query, key=key, value=key, key_padding_mask=key_mask, need_weights=False
            )[0]
        )

        if self._use_layer_norm:
            x = self._layer_norm_1(x)

        x = x + self._elementwise_transform(x)

        if self._use_layer_norm:
            x = self._layer_norm_2(x)

        return x


class SAB(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        multihead_init_type: MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: ElementwiseTransformType,
        dropout_rate: float,
    ):
        """
        Self Attention Block of the Set Transformer model.

        Args:
        :param embedding_dim: Dimension of the input data.
        :param num_heads: Number of heads.
        :param multihead_init_type: How linear layers in nn.MultiheadAttention are initialised.
        :param use_layer_norm: Whether layer normalisation should be used in SAB blocks.
        :param elementwise_transform_type: Elementwise transform (rFF) type used.
        :param dropout_rate: the dropout rate
        """
        super().__init__()
        self._mab = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Input tensor with shape ``[batch_size, set_size, embedding_dim]`` to be used as query, key, and value.
        :param mask: Boolean mask tensor with shape ``[batch_size, set_size]``, True values are masked.
                If None, everything is observed.
        Returns:
            output: Tensor with shape ``[batch_size, set_size, embedding_dim]``.
        """
        return self._mab(x, x, mask)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention block of the Set Transformer model.
    Seed vectors attend to the given values.

    :param embedding_dim: Dimension of the input data.
    :param num_heads: Number of heads.
    :param num_seed_vectors: Number of seed vectors.
    :param multihead_init_type: How linear layers in nn.MultiheadAttention are initialised.
    :param use_layer_norm: Whether layer normalisation should be used in MAB blocks.
    :param elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
    :param use_elementwise_transform_pma: Whether an elementwise transform (rFF) should be used in the PMA block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_seed_vectors: int,
        multihead_init_type: MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: ElementwiseTransformType,
        use_elementwise_transform_pma: bool,
        dropout_rate: float,
    ):
        super().__init__()
        self._mab = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
            dropout_rate=dropout_rate,
        )

        self._seed_vectors = nn.Parameter(
            torch.randn(1, num_seed_vectors, embedding_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self._seed_vectors)

        if use_elementwise_transform_pma:
            self._elementwise_transform = _create_elementwise_transform(
                embedding_dim, elementwise_transform_type
            )
        else:
            self._elementwise_transform = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape ``[batch_size, set_size, embedding_dim]`` to be used as key and value.
            mask: Mask tensor with shape ``[batch_size, set_size]``, ``True`` for masked elements.
                If ``None``, everything is observed. The mask enforces that only the selected values are
                attended to in multihead attention.
        Returns:
            Attention output tensor with shape ``[batch_size, num_seed_vectors, embedding_dim]``.
        """
        if self._elementwise_transform:
            x = self._elementwise_transform(x)

        batch_size, _, _ = x.shape
        seed_vectors_repeated = self._seed_vectors.expand(batch_size, -1, -1)
        output = self._mab(seed_vectors_repeated, x, mask)

        return output


class ISAB(nn.Module):
    """
    Inducing-point self attention block. This reduces memory use and compute time from :math:`O(N^2)` to :math:`O(NM)`
    where :math:`N` is the number of features and :math:`M` is the number of inducing points.

    Reference: https://arxiv.org/pdf/1810.00825.pdf

    Args:
        embedding_dim: Dimension of the input data.
        num_heads: Number of heads.
        num_inducing_points: Number of inducing points.
        multihead_init_type: How linear layers in nn.MultiheadAttention are initialised.
        use_layer_norm: Whether layer normalisation should be used in MAB blocks.
        elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_inducing_points: int,
        multihead_init_type: MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: ElementwiseTransformType,
        dropout_rate: float,
    ):
        super().__init__()
        self._mab1 = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
            dropout_rate=dropout_rate,
        )
        self._mab2 = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
            dropout_rate=dropout_rate,
        )
        self._inducing_points = nn.Parameter(
            torch.randn(1, num_inducing_points, embedding_dim), requires_grad=True
        )
        nn.init.xavier_uniform_(self._inducing_points)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """


        Args:
            x: Input tensor with shape ``[batch_size, set_size, embedding_dim]`` to be used as query, key and value.
            mask: Mask tensor with shape ``[batch_size, set_size]``, ``True`` values are masked.
                If ``None``, all elements are used is used. The mask enforces that only the selected
                values are attended to in multihead attention, but the output is generated for all
                elements of x.
        Returns:
            Attention output tensor with shape ``[batch_size, set_size, embedding_dim]``.
        """
        batch_size, _, _ = x.shape
        inducing_points = self._inducing_points.expand(
            batch_size, -1, -1
        )  # [batch_size, num_inducing_points, embedding_dim]
        y = self._mab1(
            query=inducing_points, key=x, key_mask=mask
        )  #  [batch_size, num_inducing_points, embedding_dim]
        return self._mab2(query=x, key=y)  # [batch_size, set_size, embedding_dim]
