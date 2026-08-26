from __future__ import annotations
from pathlib import Path
from loguru import logger
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from mlops_fraudulent_transactions.config import MLP_MODEL_PATH
from mlops_fraudulent_transactions.modeling.base import BaseModel

class MLPModel(BaseModel):
    """Supervised multi-layer perceptron for binary fraud classification."""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(input_dim)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def build(self) -> MLPModel:
        self.model = keras.Sequential(
            [
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(32, activation="relu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, activation="sigmoid"),
            ],
            name="MLP_Supervisado",
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), "accuracy"],
        )
        logger.info("MLP built and compiled.")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_path: Path = MLP_MODEL_PATH,
        epochs: int = 30,
        batch_size: int = 2048,
        validation_split: float = 0.15,
        class_weight: dict | None = None,
    ) -> keras.callbacks.History:
        if self.model is None:
            raise RuntimeError("Call build() before training.")
        callbacks = [
            EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True,
            ),
        ]
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history