"""Parameter-efficient ensemble layers used by Genie-ADLA."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _uniform_by_input_size(tensor: Tensor, input_dim: int) -> None:
    bound = 1.0 / math.sqrt(input_dim)
    nn.init.uniform_(tensor, -bound, bound)


class EnsembleView(nn.Module):
    """Expose one feature matrix as `K` zero-copy ensemble views."""

    def __init__(self, ensemble_size: int) -> None:
        super().__init__()
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.ensemble_size = ensemble_size

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must have shape (batch, features), got {tuple(x.shape)}")
        return x.unsqueeze(1).expand(-1, self.ensemble_size, -1)


class BatchEnsembleLinear(nn.Module):
    """A shared linear map with member-specific input/output scaling."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int,
        *,
        input_scale_init: str = "ones",
        output_scale_init: str = "ones",
        input_scale_chunks: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or ensemble_size <= 0:
            raise ValueError("Layer dimensions and ensemble_size must be positive")
        if input_scale_init not in {"ones", "normal"}:
            raise ValueError("input_scale_init must be 'ones' or 'normal'")
        if output_scale_init not in {"ones", "normal"}:
            raise ValueError("output_scale_init must be 'ones' or 'normal'")
        if input_scale_chunks is not None:
            if any(size <= 0 for size in input_scale_chunks):
                raise ValueError("input_scale_chunks must contain positive integers")
            if sum(input_scale_chunks) != input_dim:
                raise ValueError("input_scale_chunks must sum to input_dim")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.input_scale_init = input_scale_init
        self.output_scale_init = output_scale_init
        self.input_scale_chunks = (
            tuple(input_scale_chunks) if input_scale_chunks is not None else None
        )

        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.input_scale = nn.Parameter(torch.empty(ensemble_size, input_dim))
        self.output_scale = nn.Parameter(torch.empty(ensemble_size, output_dim))
        self.bias = nn.Parameter(torch.empty(ensemble_size, output_dim))
        self.reset_parameters()

    @staticmethod
    def _initialize_scale(
        tensor: Tensor, distribution: str, chunks: tuple[int, ...] | None = None
    ) -> None:
        if distribution == "ones":
            nn.init.ones_(tensor)
            return
        if chunks is None:
            nn.init.normal_(tensor)
            return

        with torch.no_grad():
            start = 0
            for size in chunks:
                values = torch.randn(
                    tensor.shape[0], 1, device=tensor.device, dtype=tensor.dtype
                )
                tensor[:, start : start + size] = values
                start += size

    def reset_parameters(self) -> None:
        _uniform_by_input_size(self.weight, self.input_dim)
        self._initialize_scale(
            self.input_scale, self.input_scale_init, self.input_scale_chunks
        )
        self._initialize_scale(self.output_scale, self.output_scale_init)
        shared_bias = torch.empty(
            self.output_dim, device=self.weight.device, dtype=self.weight.dtype
        )
        _uniform_by_input_size(shared_bias, self.input_dim)
        with torch.no_grad():
            self.bias.copy_(shared_bias.unsqueeze(0).expand_as(self.bias))

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.ensemble_size, self.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "x must have shape "
                f"(batch, {self.ensemble_size}, {self.input_dim}), got {tuple(x.shape)}"
            )
        x = x * self.input_scale
        x = torch.matmul(x, self.weight.T)
        return x * self.output_scale + self.bias


class EnsembleLinear(nn.Module):
    """Independent linear output heads for every ensemble member."""

    def __init__(
        self, input_dim: int, output_dim: int, ensemble_size: int
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or ensemble_size <= 0:
            raise ValueError("Layer dimensions and ensemble_size must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.empty(ensemble_size, input_dim, output_dim)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _uniform_by_input_size(self.weight, self.input_dim)
        _uniform_by_input_size(self.bias, self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.ensemble_size, self.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "x must have shape "
                f"(batch, {self.ensemble_size}, {self.input_dim}), got {tuple(x.shape)}"
            )
        return torch.einsum("bki,kio->bko", x, self.weight) + self.bias


class TabMBlock(nn.Module):
    """BatchEnsemble linear transformation followed by activation and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int,
        dropout: float,
        activation: str,
        *,
        first_block: bool,
        input_scale_chunks: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation == "relu":
            activation_module: nn.Module = nn.ReLU()
        elif activation == "gelu":
            activation_module = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.linear = BatchEnsembleLinear(
            input_dim,
            output_dim,
            ensemble_size,
            input_scale_init="normal" if first_block else "ones",
            output_scale_init="ones",
            input_scale_chunks=input_scale_chunks if first_block else None,
        )
        self.activation = activation_module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))

