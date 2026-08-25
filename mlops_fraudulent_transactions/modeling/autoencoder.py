from __future__ import annotations
from pathlib import Path
from loguru import logger
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from mlops_fraudulent_transactions.config import AUTOENCODER_MODEL_PATH
from mlops_fraudulent_transactions.modeling.base import BaseModel

class AutoencoderModel(BaseModel):
    """Autoencoder that reconstructs normal transactions to flag anomalies."""

    def __init__(self, input_dim: int, encoding_dim: int = 14) -> None:
        super().__init__(input_dim)
        self.encoding_dim = encoding_dim

    def build(self) -> AutoencoderModel:
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(20, activation="tanh")(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)
        decoded = layers.Dense(20, activation="tanh")(encoded)
        decoded = layers.Dense(self.input_dim, activation="linear")(decoded)

        self.model = keras.Model(inputs=input_layer, outputs=decoded, name="Autoencoder_Anomalias")
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        logger.info("Autoencoder built and compiled.")
        return self

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        checkpoint_path: Path = AUTOENCODER_MODEL_PATH,
        epochs: int = 30,
        batch_size: int = 2048,
        validation_split: float = 0.15,
    ) -> keras.callbacks.History:
        if self.model is None:
            raise RuntimeError("Call build() before training.")
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True),
            ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            ),
        ]
        self.history = self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )
        return self.history

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return the normalized MSE reconstruction error as anomaly score."""
        reconstructed = self.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        mse_min = mse.min()
        mse_max = mse.max()
        return (mse - mse_min) / (mse_max - mse_min)

    def anomaly_threshold(
        self,
        x_normal: np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """Derive an anomaly threshold from the errors of normal transactions."""
        errors = self.reconstruction_error(x_normal)
        return float(np.percentile(errors, percentile))