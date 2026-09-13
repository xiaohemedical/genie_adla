"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    test_path: str
    label_column: str = "label"
    index_column: str = "sample_id"
    validation_fraction: float = 0.2
    split_seed: int = 42


@dataclass(frozen=True)
class PreprocessingConfig:
    n_quantiles: int = 1000
    output_distribution: str = "normal"
    n_bins: int = 48


@dataclass(frozen=True)
class ModelConfig:
    ensemble_size: int = 32
    n_blocks: int = 2
    hidden_dim: int = 512
    embedding_dim: int = 16
    dropout: float = 0.1
    activation: str = "relu"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    batch_size: int = 256
    eval_batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    early_stopping_patience: int = 20
    gradient_clip_norm: float | None = 1.0
    device: str = "auto"


@dataclass(frozen=True)
class EvaluationConfig:
    target_specificities: tuple[float, ...] = (0.95, 0.99)
    threshold_source: str = "validation"


@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool = False
    n_splits: int = 5
    shuffle: bool = True
    split_seed: int = 42


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs"


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    cross_validation: CrossValidationConfig
    output: OutputConfig


_T = TypeVar("_T")


def _build_section(section_type: type[_T], raw: Any, name: str) -> _T:
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    allowed = {field.name for field in fields(section_type)}
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in config section '{name}': {names}")
    try:
        return section_type(**raw)
    except TypeError as error:
        raise ValueError(f"Invalid config section '{name}': {error}") from error


def load_config(path: str | Path) -> Config:
    """Load, construct, and validate a YAML configuration."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("The root of the config must be a mapping")

    required = {
        "seed",
        "data",
        "preprocessing",
        "model",
        "training",
        "evaluation",
        "output",
    }
    allowed = required | {"cross_validation"}
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown top-level config keys: {names}")

    missing = required - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing top-level config sections: {names}")

    evaluation_raw = dict(raw["evaluation"])
    if "target_specificities" in evaluation_raw:
        evaluation_raw["target_specificities"] = tuple(
            evaluation_raw["target_specificities"]
        )

    config = Config(
        seed=int(raw["seed"]),
        data=_build_section(DataConfig, raw["data"], "data"),
        preprocessing=_build_section(
            PreprocessingConfig, raw["preprocessing"], "preprocessing"
        ),
        model=_build_section(ModelConfig, raw["model"], "model"),
        training=_build_section(TrainingConfig, raw["training"], "training"),
        evaluation=_build_section(EvaluationConfig, evaluation_raw, "evaluation"),
        cross_validation=_build_section(
            CrossValidationConfig,
            raw.get("cross_validation", {}),
            "cross_validation",
        ),
        output=_build_section(OutputConfig, raw["output"], "output"),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Validate values that cannot be checked by dataclass construction."""
    if not 0.0 < config.data.validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be in (0, 1)")
    if config.preprocessing.n_quantiles < 2:
        raise ValueError("preprocessing.n_quantiles must be at least 2")
    if config.preprocessing.n_bins < 2:
        raise ValueError("preprocessing.n_bins must be at least 2")
    if config.preprocessing.output_distribution not in {"normal", "uniform"}:
        raise ValueError("preprocessing.output_distribution must be 'normal' or 'uniform'")
    if config.model.ensemble_size <= 0:
        raise ValueError("model.ensemble_size must be positive")
    if config.model.n_blocks <= 0:
        raise ValueError("model.n_blocks must be positive")
    if config.model.hidden_dim <= 0 or config.model.embedding_dim <= 0:
        raise ValueError("model dimensions must be positive")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.model.activation not in {"relu", "gelu"}:
        raise ValueError("model.activation must be 'relu' or 'gelu'")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if config.training.batch_size <= 0 or config.training.eval_batch_size <= 0:
        raise ValueError("training batch sizes must be positive")
    if config.training.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if not 0.0 < config.training.scheduler_factor < 1.0:
        raise ValueError("training.scheduler_factor must be in (0, 1)")
    if config.training.scheduler_patience < 0:
        raise ValueError("training.scheduler_patience must be non-negative")
    if config.training.early_stopping_patience < 0:
        raise ValueError("training.early_stopping_patience must be non-negative")
    if config.training.gradient_clip_norm is not None:
        if config.training.gradient_clip_norm <= 0.0:
            raise ValueError("training.gradient_clip_norm must be positive or null")
    if config.training.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be 'auto', 'cpu', or 'cuda'")
    if not config.evaluation.target_specificities:
        raise ValueError("evaluation.target_specificities must not be empty")
    if any(not 0.0 < value < 1.0 for value in config.evaluation.target_specificities):
        raise ValueError("Each target specificity must be in (0, 1)")
    if config.evaluation.threshold_source not in {"validation", "test"}:
        raise ValueError(
            "evaluation.threshold_source must be 'validation' or 'test'"
        )
    if config.cross_validation.n_splits < 2:
        raise ValueError("cross_validation.n_splits must be at least 2")


def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a config to a YAML/JSON-serializable mapping."""
    result = asdict(config)
    result["evaluation"]["target_specificities"] = list(
        config.evaluation.target_specificities
    )
    return result


def save_config(config: Config, path: str | Path) -> None:
    """Save the resolved configuration."""
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(config_to_dict(config), file, sort_keys=False)
