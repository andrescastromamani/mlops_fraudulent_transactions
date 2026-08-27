from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow import keras


class BaseModel(ABC):
    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.model: keras.Model | None = None
        self.history: keras.callbacks.History | None = None

    @abstractmethod
    def build(self) -> BaseModel:
        """Construct and compile the underlying Keras model."""

    @abstractmethod
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray | None = None,
        checkpoint_path: Path = Path(""),
        epochs: int = 30,
        batch_size: int = 2048,
    ) -> keras.callbacks.History:
        """Train the model and return the training history."""

    def save(self, checkpoint_path: Path) -> None:
        """Serialize the model to disk."""
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        self.model.save(checkpoint_path)
        logger.success(f"Model saved to {checkpoint_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the raw model output for the given features."""
        if self.model is None:
            raise RuntimeError("Model not built yet.")
        return self.model.predict(X, batch_size=2048)
