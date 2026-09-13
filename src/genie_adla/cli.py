"""Command-line orchestration for single-split and cross-validation training."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .config import Config, load_config, save_config
from .data import (
    DataBundle,
    DatasetSplit,
    load_data,
    load_full_data,
    select_rows,
)
from .evaluation import (
    compute_auc,
    evaluate_at_thresholds,
    positive_class_scores,
    predict_logits,
    select_specificity_thresholds,
)
from .model import TabM
from .preprocessing import (
    NumericalPreprocessor,
    compute_quantile_bins,
    transform_data,
)
from .training import resolve_device, set_random_seed, train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Genie-ADLA on numerical features for binary classification."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser.parse_args()


def _prediction_frame(split: DatasetSplit, scores: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": split.sample_ids,
            "label": split.y,
            "proba_case": scores,
            "prediction": (scores > 0.5).astype(np.int64),
        }
    )


def _make_model(config: Config, bins: list[np.ndarray]) -> TabM:
    return TabM(
        bins,
        embedding_dim=config.model.embedding_dim,
        ensemble_size=config.model.ensemble_size,
        n_blocks=config.model.n_blocks,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        activation=config.model.activation,
    )


def _create_run_directory(root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = Path(root) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _save_model(model: TabM, config: Config, path: Path) -> None:
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "bins": model.export_bins(),
            "model_config": asdict(config.model),
        },
        path,
    )


def _environment_metadata(
    config: Config, feature_names: list[str], device: torch.device
) -> dict[str, Any]:
    return {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "device": str(device),
        "feature_names": feature_names,
        "label_mapping": {"negative": 0, "positive": 1},
        "configured_threshold_source": config.evaluation.threshold_source,
    }


def _write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def _threshold_results(
    config: Config,
    validation_y: np.ndarray,
    validation_scores: np.ndarray,
    test_data: DatasetSplit,
    test_scores: np.ndarray,
    *,
    validation_source_name: str,
) -> tuple[str, dict[str, float], dict[str, dict[str, float | int]]]:
    if config.evaluation.threshold_source == "validation":
        source_name = validation_source_name
        reference_y = validation_y
        reference_scores = validation_scores
    else:
        source_name = "test"
        reference_y = test_data.y
        reference_scores = test_scores
    thresholds = select_specificity_thresholds(
        reference_y,
        reference_scores,
        config.evaluation.target_specificities,
    )
    threshold_metrics = evaluate_at_thresholds(
        test_data.y, test_scores, thresholds
    )
    return source_name, thresholds, threshold_metrics


def _run_single_split(config: Config, device: torch.device) -> Path:
    raw_data = load_data(config.data)
    preprocessor = NumericalPreprocessor(
        config.preprocessing, random_state=config.seed
    ).fit(raw_data.train.x)
    data = transform_data(raw_data, preprocessor)
    bins = compute_quantile_bins(data.train.x, config.preprocessing.n_bins)
    model = _make_model(config, bins)

    result = train_model(
        model,
        data.train,
        data.validation,
        config.training,
        device=device,
        seed=config.seed,
    )
    validation_scores = positive_class_scores(
        predict_logits(
            model,
            data.validation.x,
            device=device,
            batch_size=config.training.eval_batch_size,
        )
    )
    test_scores = positive_class_scores(
        predict_logits(
            model,
            data.test.x,
            device=device,
            batch_size=config.training.eval_batch_size,
        )
    )
    threshold_source, thresholds, threshold_metrics = _threshold_results(
        config,
        data.validation.y,
        validation_scores,
        data.test,
        test_scores,
        validation_source_name="validation",
    )

    metrics: dict[str, Any] = {
        "mode": "single_split",
        "best_epoch": result.best_epoch,
        "best_validation_auc": result.best_validation_auc,
        "validation_auc": compute_auc(data.validation.y, validation_scores),
        "test_auc": compute_auc(data.test.y, test_scores),
        "threshold_source": threshold_source,
        "threshold_metrics": threshold_metrics,
        "history": result.history,
    }
    metadata = _environment_metadata(config, raw_data.feature_names, device)
    metadata.update(
        {
            "mode": "single_split",
            "best_epoch": result.best_epoch,
            "threshold_source": threshold_source,
            "thresholds": thresholds,
        }
    )

    run_directory = _create_run_directory(config.output.directory)
    save_config(config, run_directory / "config.yaml")
    preprocessor.save(run_directory / "preprocessor.joblib")
    _save_model(model, config, run_directory / "model.pt")
    _prediction_frame(data.validation, validation_scores).to_csv(
        run_directory / "validation_predictions.csv", index=False
    )
    _prediction_frame(data.test, test_scores).to_csv(
        run_directory / "test_predictions.csv", index=False
    )
    _write_json(metrics, run_directory / "metrics.json")
    _write_json(metadata, run_directory / "metadata.json")
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "mode": "single_split",
                "validation_auc": metrics["validation_auc"],
                "test_auc": metrics["test_auc"],
                "threshold_source": threshold_source,
            },
            indent=2,
        )
    )
    return run_directory


def _run_cross_validation(config: Config, device: torch.device) -> Path:
    full_data = load_full_data(config.data)
    n_splits = config.cross_validation.n_splits
    class_counts = np.bincount(full_data.train.y, minlength=2)
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            "Each class must contain at least cross_validation.n_splits samples"
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config.cross_validation.shuffle,
        random_state=(
            config.cross_validation.split_seed
            if config.cross_validation.shuffle
            else None
        ),
    )
    oof_scores = np.full(len(full_data.train.y), np.nan, dtype=np.float32)
    oof_folds = np.full(len(full_data.train.y), -1, dtype=np.int64)
    fold_test_scores: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []

    run_directory = _create_run_directory(config.output.directory)
    save_config(config, run_directory / "config.yaml")

    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(full_data.train.x, full_data.train.y), start=1
    ):
        print(f"\n===== fold {fold_index}/{n_splits} =====")
        fold_seed = config.seed + fold_index - 1
        set_random_seed(fold_seed)
        fold_raw = DataBundle(
            train=select_rows(full_data.train, train_indices),
            validation=select_rows(full_data.train, validation_indices),
            test=full_data.test,
            feature_names=full_data.feature_names,
        )
        preprocessor = NumericalPreprocessor(
            config.preprocessing, random_state=fold_seed
        ).fit(fold_raw.train.x)
        fold_data = transform_data(fold_raw, preprocessor)
        bins = compute_quantile_bins(
            fold_data.train.x, config.preprocessing.n_bins
        )
        model = _make_model(config, bins)
        result = train_model(
            model,
            fold_data.train,
            fold_data.validation,
            config.training,
            device=device,
            seed=fold_seed,
        )
        validation_scores = positive_class_scores(
            predict_logits(
                model,
                fold_data.validation.x,
                device=device,
                batch_size=config.training.eval_batch_size,
            )
        )
        test_scores = positive_class_scores(
            predict_logits(
                model,
                fold_data.test.x,
                device=device,
                batch_size=config.training.eval_batch_size,
            )
        )
        oof_scores[validation_indices] = validation_scores
        oof_folds[validation_indices] = fold_index
        fold_test_scores.append(test_scores)

        fold_directory = run_directory / f"fold_{fold_index:02d}"
        fold_directory.mkdir()
        preprocessor.save(fold_directory / "preprocessor.joblib")
        _save_model(model, config, fold_directory / "model.pt")
        fold_frame = _prediction_frame(
            fold_data.validation, validation_scores
        )
        fold_frame.insert(1, "fold", fold_index)
        fold_frame.to_csv(
            fold_directory / "validation_predictions.csv", index=False
        )
        fold_summaries.append(
            {
                "fold": fold_index,
                "seed": fold_seed,
                "best_epoch": result.best_epoch,
                "best_validation_auc": result.best_validation_auc,
                "history": result.history,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.isnan(oof_scores).any() or np.any(oof_folds < 1):
        raise RuntimeError("Cross-validation did not produce exactly one OOF prediction per sample")
    mean_test_scores = np.mean(np.stack(fold_test_scores, axis=0), axis=0)
    threshold_source, thresholds, threshold_metrics = _threshold_results(
        config,
        full_data.train.y,
        oof_scores,
        full_data.test,
        mean_test_scores,
        validation_source_name="out_of_fold",
    )
    oof_auc = compute_auc(full_data.train.y, oof_scores)
    test_auc = compute_auc(full_data.test.y, mean_test_scores)
    metrics: dict[str, Any] = {
        "mode": "cross_validation",
        "n_splits": n_splits,
        "oof_auc": oof_auc,
        "test_auc": test_auc,
        "threshold_source": threshold_source,
        "threshold_metrics": threshold_metrics,
        "folds": fold_summaries,
    }
    metadata = _environment_metadata(config, full_data.feature_names, device)
    metadata.update(
        {
            "mode": "cross_validation",
            "n_splits": n_splits,
            "threshold_source": threshold_source,
            "thresholds": thresholds,
        }
    )

    oof_frame = _prediction_frame(full_data.train, oof_scores)
    oof_frame.insert(1, "fold", oof_folds)
    oof_frame.to_csv(run_directory / "oof_predictions.csv", index=False)
    _prediction_frame(full_data.test, mean_test_scores).to_csv(
        run_directory / "test_predictions.csv", index=False
    )
    _write_json(metrics, run_directory / "metrics.json")
    _write_json(metadata, run_directory / "metadata.json")
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "mode": "cross_validation",
                "n_splits": n_splits,
                "oof_auc": oof_auc,
                "test_auc": test_auc,
                "threshold_source": threshold_source,
            },
            indent=2,
        )
    )
    return run_directory


def run(config: Config) -> Path:
    """Execute one single-split or cross-validation training run."""
    set_random_seed(config.seed)
    device = resolve_device(config.training.device)
    if config.cross_validation.enabled:
        return _run_cross_validation(config, device)
    return _run_single_split(config, device)


def main() -> None:
    args = _parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
