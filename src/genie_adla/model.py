"""The numerical-feature binary-classification Genie-ADLA model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .embeddings import PiecewiseLinearEmbedding
from .layers import EnsembleLinear, EnsembleView, TabMBlock


class TabM(nn.Module):
    """The Genie-ADLA numerical classifier returning `(B, K, 2)` logits."""

    def __init__(
        self,
        bins: Sequence[np.ndarray | Tensor],
        *,
        embedding_dim: int,
        ensemble_size: int,
        n_blocks: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.ensemble_size = ensemble_size
        self.num_embeddings = PiecewiseLinearEmbedding(
            bins, embedding_dim=embedding_dim, activation=activation
        )
        self.ensemble_view = EnsembleView(ensemble_size)

        input_dim = self.num_embeddings.output_dim
        chunks = [embedding_dim] * self.num_embeddings.n_features
        blocks: list[nn.Module] = []
        for block_index in range(n_blocks):
            blocks.append(
                TabMBlock(
                    input_dim if block_index == 0 else hidden_dim,
                    hidden_dim,
                    ensemble_size,
                    dropout,
                    activation,
                    first_block=block_index == 0,
                    input_scale_chunks=chunks if block_index == 0 else None,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output = EnsembleLinear(hidden_dim, 2, ensemble_size)

    @property
    def n_features(self) -> int:
        return self.num_embeddings.n_features

    def export_bins(self) -> list[Tensor]:
        return self.num_embeddings.encoder.export_bins()

    def forward(self, x_num: Tensor) -> Tensor:
        if x_num.ndim != 2:
            raise ValueError(
                f"x_num must have shape (batch, features), got {tuple(x_num.shape)}"
            )
        if x_num.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} numerical features, got {x_num.shape[1]}"
            )

        x = self.num_embeddings(x_num).flatten(1)
        x = self.ensemble_view(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def load_model(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> TabM:
    """Reconstruct a model from a Genie-ADLA checkpoint."""
    checkpoint: dict[str, Any] = torch.load(
        Path(path), map_location=map_location, weights_only=True
    )
    required = {"state_dict", "bins", "model_config"}
    missing = required - set(checkpoint)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Invalid model checkpoint; missing keys: {names}")
    model = TabM(checkpoint["bins"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
