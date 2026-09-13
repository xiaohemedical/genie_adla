"""Piecewise-linear encodings and learnable numerical embeddings."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _prepare_bins(
    bins: Sequence[np.ndarray | Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if not bins:
        raise ValueError("bins must contain at least one feature")

    tensors: list[Tensor] = []
    for index, values in enumerate(bins):
        edges = torch.as_tensor(values, dtype=torch.float32)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                f"bins[{index}] must be one-dimensional with at least two edges"
            )
        if not torch.isfinite(edges).all() or not torch.all(edges[1:] > edges[:-1]):
            raise ValueError(
                f"bins[{index}] must contain finite, strictly increasing edges"
            )
        tensors.append(edges)

    interval_counts = torch.tensor(
        [len(edges) - 1 for edges in tensors], dtype=torch.long
    )
    max_intervals = int(interval_counts.max())
    padded_edges = torch.empty((len(tensors), max_intervals + 1))
    interval_mask = torch.zeros((len(tensors), max_intervals), dtype=torch.bool)
    for feature_index, edges in enumerate(tensors):
        count = len(edges) - 1
        padded_edges[feature_index, : len(edges)] = edges
        padded_edges[feature_index, len(edges) :] = edges[-1]
        interval_mask[feature_index, :count] = True
    return padded_edges, interval_mask, interval_counts


class PiecewiseLinearEncoder(nn.Module):
    """Encode each scalar by its progress through ordered feature-specific bins.

    For an interval `[left, right]`, its component is
    `clip((x - left) / (right - left), 0, 1)`. Consequently, intervals below
    the value are encoded as one, the active interval is fractional, and intervals
    above the value are encoded as zero.
    """

    def __init__(self, bins: Sequence[np.ndarray | Tensor]) -> None:
        super().__init__()
        edges, mask, counts = _prepare_bins(bins)
        self.register_buffer("edges", edges)
        self.register_buffer("interval_mask", mask)
        self.register_buffer("interval_counts", counts)

    @property
    def n_features(self) -> int:
        return self.edges.shape[0]

    @property
    def max_intervals(self) -> int:
        return self.interval_mask.shape[1]

    def export_bins(self) -> list[Tensor]:
        """Return unpadded bin edges on CPU for checkpoint reconstruction."""
        return [
            self.edges[index, : int(count) + 1].detach().cpu().clone()
            for index, count in enumerate(self.interval_counts.tolist())
        ]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, features), got {tuple(x.shape)}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        left = self.edges[:, :-1]
        right = self.edges[:, 1:]
        widths = right - left
        safe_widths = torch.where(self.interval_mask, widths, torch.ones_like(widths))
        encoded = ((x.unsqueeze(-1) - left) / safe_widths).clamp(0.0, 1.0)
        return encoded.masked_fill(~self.interval_mask, 0.0)


class PiecewiseLinearEmbedding(nn.Module):
    """Apply a learnable feature-wise projection to piecewise-linear encodings."""

    def __init__(
        self,
        bins: Sequence[np.ndarray | Tensor],
        embedding_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.encoder = PiecewiseLinearEncoder(bins)
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.weight = nn.Parameter(
            torch.empty(
                self.encoder.n_features,
                self.encoder.max_intervals,
                embedding_dim,
            )
        )
        self.bias = nn.Parameter(
            torch.empty(self.encoder.n_features, embedding_dim)
        )
        self.reset_parameters()

    @property
    def n_features(self) -> int:
        return self.encoder.n_features

    @property
    def output_dim(self) -> int:
        return self.n_features * self.embedding_dim

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            for feature_index, count in enumerate(
                self.encoder.interval_counts.tolist()
            ):
                bound = 1.0 / math.sqrt(count)
                self.weight[feature_index, :count].uniform_(-bound, bound)
                self.bias[feature_index].uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        embedded = torch.einsum("bfp,fpe->bfe", encoded, self.weight) + self.bias
        if self.activation == "relu":
            return F.relu(embedded)
        return F.gelu(embedded)

