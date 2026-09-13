"""CSV loading and a single stratified train/validation split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig


@dataclass(frozen=True)
class DatasetSplit:
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class DataBundle:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    feature_names: list[str]


@dataclass(frozen=True)
class FullData:
    train: DatasetSplit
    test: DatasetSplit
    feature_names: list[str]


@dataclass(frozen=True)
class _LoadedTable:
    x: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    feature_names: list[str]


def _load_table(path: str | Path, label_column: str, index_column: str) -> _LoadedTable:
    path = Path(path)
    table = pd.read_csv(path)
    required = {index_column, label_column}
    missing = required - set(table.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {path}: {names}")

    if table[index_column].isna().any():
        raise ValueError(f"Column '{index_column}' contains missing sample IDs in {path}")
    if table[index_column].duplicated().any():
        raise ValueError(f"Column '{index_column}' contains duplicate sample IDs in {path}")

    feature_names = [
        name for name in table.columns if name not in {index_column, label_column}
    ]
    if not feature_names:
        raise ValueError(f"No feature columns found in {path}")

    non_numeric = [
        name
        for name in feature_names
        if not pd.api.types.is_numeric_dtype(table[name].dtype)
    ]
    if non_numeric:
        names = ", ".join(non_numeric)
        raise ValueError(f"All features must be numerical; invalid columns in {path}: {names}")

    if table[label_column].isna().any():
        raise ValueError(f"Column '{label_column}' contains missing labels in {path}")
    label_values = set(table[label_column].unique().tolist())
    if not label_values.issubset({0, 1}) or label_values != {0, 1}:
        raise ValueError(
            f"Column '{label_column}' in {path} must contain both binary labels 0 and 1"
        )

    x = table[feature_names].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError(f"Feature matrix in {path} contains NaN or infinite values")

    return _LoadedTable(
        x=x,
        y=table[label_column].to_numpy(dtype=np.int64, copy=True),
        sample_ids=table[index_column].astype(str).to_numpy(copy=True),
        feature_names=feature_names,
    )


def load_full_data(config: DataConfig) -> FullData:
    """Load and validate complete train and test tables without splitting."""
    train_table = _load_table(
        config.train_path, config.label_column, config.index_column
    )
    test_table = _load_table(config.test_path, config.label_column, config.index_column)

    if train_table.feature_names != test_table.feature_names:
        train_set = set(train_table.feature_names)
        test_set = set(test_table.feature_names)
        missing = sorted(train_set - test_set)
        unexpected = sorted(test_set - train_set)
        if not missing and not unexpected:
            detail = "the columns have a different order"
        else:
            detail = f"missing={missing}, unexpected={unexpected}"
        raise ValueError(f"Train/test feature columns do not match: {detail}")

    return FullData(
        train=DatasetSplit(
            train_table.x, train_table.y, train_table.sample_ids
        ),
        test=DatasetSplit(test_table.x, test_table.y, test_table.sample_ids),
        feature_names=train_table.feature_names,
    )


def select_rows(data: DatasetSplit, indices: np.ndarray) -> DatasetSplit:
    """Select samples without changing their order inside ``indices``."""
    return DatasetSplit(
        x=data.x[indices],
        y=data.y[indices],
        sample_ids=data.sample_ids[indices],
    )


def load_data(config: DataConfig) -> DataBundle:
    """Load CSV files and split the training table once with stratification."""
    full_data = load_full_data(config)

    indices = np.arange(len(full_data.train.y))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_fraction,
        random_state=config.split_seed,
        stratify=full_data.train.y,
    )

    return DataBundle(
        train=select_rows(full_data.train, train_indices),
        validation=select_rows(full_data.train, validation_indices),
        test=full_data.test,
        feature_names=full_data.feature_names,
    )
