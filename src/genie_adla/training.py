"""Training loop and validation-based model selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig
from .data import DatasetSplit
from .evaluation import compute_auc, positive_class_scores, predict_logits


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_auc: float
    history: list[dict[str, float | int]]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def ensemble_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Supervise every ensemble member with the same binary labels."""
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape (batch, ensemble, 2)")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must have shape (batch,)")
    ensemble_size = logits.shape[1]
    flat_logits = logits.reshape(-1, 2)
    flat_labels = labels.repeat_interleave(ensemble_size)
    return nn.functional.cross_entropy(flat_logits, flat_labels)


def train_model(
    model: nn.Module,
    train_data: DatasetSplit,
    validation_data: DatasetSplit,
    config: TrainingConfig,
    *,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    """Train a model and restore the state with the best validation AUC."""
    model.to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(train_data.x), torch.from_numpy(train_data.y)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_auc = -float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ensemble_cross_entropy(model(x_batch), y_batch)
            loss.backward()
            if config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            batch_samples = len(x_batch)
            total_loss += float(loss.detach()) * batch_samples
            total_samples += batch_samples

        validation_logits = predict_logits(
            model,
            validation_data.x,
            device=device,
            batch_size=config.eval_batch_size,
        )
        validation_score = positive_class_scores(validation_logits)
        validation_auc = compute_auc(validation_data.y, validation_score)
        scheduler.step(validation_auc)
        mean_loss = total_loss / total_samples
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_auc": validation_auc,
                "learning_rate": learning_rate,
            }
        )

        improved = validation_auc > best_auc
        marker = "*" if improved else " "
        print(
            f"{marker} epoch={epoch:03d} train_loss={mean_loss:.6f} "
            f"validation_auc={validation_auc:.6f} lr={learning_rate:.3g}"
        )
        if improved:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_auc=best_auc,
        history=history,
    )
