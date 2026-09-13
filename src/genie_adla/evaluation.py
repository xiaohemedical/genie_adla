"""Prediction, ensemble aggregation, and binary classification metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import Tensor, nn


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Run batched inference and return CPU logits."""
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional NumPy array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    logits: list[Tensor] = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start : start + batch_size], device=device)
        logits.append(model(batch).detach().cpu())
    if not logits:
        raise ValueError("x must contain at least one sample")
    return torch.cat(logits, dim=0)


def ensemble_probabilities(logits: Tensor) -> Tensor:
    """Average member probabilities from logits with shape `(B, K, 2)`."""
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(
            f"logits must have shape (batch, ensemble, 2), got {tuple(logits.shape)}"
        )
    return torch.softmax(logits, dim=-1).mean(dim=1)


def positive_class_scores(logits: Tensor) -> np.ndarray:
    """Return the ensemble-mean positive-class probabilities as a NumPy array."""
    return ensemble_probabilities(logits)[:, 1].detach().cpu().numpy()


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    _validate_binary_arrays(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def select_specificity_thresholds(
    reference_y: np.ndarray,
    reference_score: np.ndarray,
    target_specificities: Iterable[float],
) -> dict[str, float]:
    """Select strict `score > threshold` rules from reference negative scores."""
    reference_y = np.asarray(reference_y)
    reference_score = np.asarray(reference_score)
    _validate_binary_arrays(reference_y, reference_score)
    negative_scores = reference_score[reference_y == 0]
    thresholds: dict[str, float] = {}
    for target in target_specificities:
        target = float(target)
        if not 0.0 < target < 1.0:
            raise ValueError("Each target specificity must be in (0, 1)")
        thresholds[_specificity_key(target)] = float(
            np.quantile(negative_scores, target, method="higher")
        )
    return thresholds


def evaluate_at_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    """Evaluate fixed thresholds without estimating anything from the test set."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    _validate_binary_arrays(y_true, y_score)
    results: dict[str, dict[str, float | int]] = {}
    for key, threshold in thresholds.items():
        prediction = (y_score > threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()
        specificity = tn / (tn + fp) if tn + fp else 0.0
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        results[key] = {
            "threshold": float(threshold),
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    return results


def _validate_binary_arrays(y_true: np.ndarray, y_score: np.ndarray) -> None:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length")
    if len(y_true) == 0:
        raise ValueError("Metric inputs must not be empty")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("y_true must contain both binary classes 0 and 1")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains NaN or infinite values")


def _specificity_key(value: float) -> str:
    percentage = value * 100.0
    if percentage.is_integer():
        return f"specificity_{int(percentage)}"
    return f"specificity_{percentage:g}"
