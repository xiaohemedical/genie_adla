"""Genie-ADLA: deep learning for high-dimensional DNA methylation classification."""

from .model import TabM, load_model

__all__ = ["TabM", "load_model"]
__version__ = "0.1.0"
