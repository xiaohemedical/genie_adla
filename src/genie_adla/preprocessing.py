"""Train-only numerical preprocessing and bin estimation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from .config import PreprocessingConfig
from .data import DataBundle, DatasetSplit


class NumericalPreprocessor:
    """A small stateful wrapper around scikit-learn's QuantileTransformer."""

    def __init__(self, config: PreprocessingConfig, random_state: int) -> None:
        self.config = config
        self.random_state = random_state
        self.transformer: QuantileTransformer | None = None

    def fit(self, x_train: np.ndarray) -> "NumericalPreprocessor":
        _validate_matrix(x_train, "x_train")
        n_quantiles = min(self.config.n_quantiles, len(x_train))
        self.transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=self.config.output_distribution,
            subsample=10**9,
            random_state=self.random_state,
        )
        self.transformer.fit(x_train)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        _validate_matrix(x, "x")
        if self.transformer is None:
            raise RuntimeError("NumericalPreprocessor must be fitted before transform")
        transformed = self.transformer.transform(x)
        return np.asarray(transformed, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        if self.transformer is None:
            raise RuntimeError("Cannot save an unfitted NumericalPreprocessor")
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> "NumericalPreprocessor":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, NumericalPreprocessor):
            raise TypeError("The loaded object is not a NumericalPreprocessor")
        return loaded


def _validate_matrix(x: np.ndarray, name: str) -> None:
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional NumPy array")
    if len(x) == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must have at least one sample and one feature")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def transform_data(
    data: DataBundle, preprocessor: NumericalPreprocessor
) -> DataBundle:
    """Return a new bundle with transformed features and unchanged metadata."""

    def transform_split(split: DatasetSplit) -> DatasetSplit:
        return replace(split, x=preprocessor.transform(split.x))

    return DataBundle(
        train=transform_split(data.train),
        validation=transform_split(data.validation),
        test=transform_split(data.test),
        feature_names=list(data.feature_names),
    )


def compute_quantile_bins(x_train: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Compute at most ``n_bins`` quantile intervals for every feature.

    Duplicate quantile edges are removed. Constant features are rejected because a
    piecewise-linear encoding requires at least one interval.
    """
    _validate_matrix(x_train, "x_train")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    result: list[np.ndarray] = []
    constant_features: list[int] = []
    for feature_index in range(x_train.shape[1]):
        edges = np.quantile(
            x_train[:, feature_index], quantiles, method="linear"
        ).astype(np.float32)
        edges = np.unique(edges)
        if len(edges) < 2:
            constant_features.append(feature_index)
        else:
            result.append(edges)

    if constant_features:
        indices = ", ".join(map(str, constant_features[:10]))
        suffix = "..." if len(constant_features) > 10 else ""
        raise ValueError(
            "Piecewise-linear bins cannot be computed for constant feature indices: "
            f"{indices}{suffix}"
        )
    return result


def validate_bins(bins: Sequence[np.ndarray]) -> None:
    """Validate externally loaded bins before constructing the model."""
    if not bins:
        raise ValueError("bins must contain at least one feature")
    for index, edges in enumerate(bins):
        edges = np.asarray(edges)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(f"bins[{index}] must be one-dimensional with at least two edges")
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError(f"bins[{index}] must contain finite, strictly increasing edges")
