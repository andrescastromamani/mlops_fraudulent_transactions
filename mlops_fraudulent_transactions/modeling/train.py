import yaml
from pathlib import Path

import numpy as np
from loguru import logger
import pandas as pd
import mlflow
import mlflow.keras
import typer

from mlops_fraudulent_transactions.config import (
    PROCESSED_DATA_DIR,
    MLP_MODEL_PATH,
    AUTOENCODER_MODEL_PATH,
)
from mlops_fraudulent_transactions.modeling import AutoencoderModel, MLPModel
app = typer.Typer()


@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    train_labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    mlp_path: Path = MLP_MODEL_PATH,
    autoencoder_path: Path = AUTOENCODER_MODEL_PATH,
    params_path: Path = Path("params.yaml"),
) -> None:
    # 1. Definir el experimento de MLflow
    mlflow.set_experiment("Fraud_Detection")

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    learning_rate = params["train"]["learning_rate"]

    logger.info(f"Cargando datos desde {train_features_path}...")
    X_train = pd.read_csv(train_features_path).to_numpy()
    y_train = pd.read_csv(train_labels_path).to_numpy().ravel()
    input_dim = X_train.shape[1]

    # 2. Iniciar el registro de la corrida (Run)
    with mlflow.start_run(run_name="Train_MLP_and_Autoencoder"):
        # Registrar parámetros de params.yaml
        mlflow.log_params(params["prepare"])
        mlflow.log_params(params["train"])

        # Callback automático para métricas por epoch en Keras
        keras_callback = mlflow.keras.MLflowCallback()

        # --- MLP ---
        logger.info("Entrenando MLP...")
        mlp = MLPModel(input_dim=input_dim, learning_rate=learning_rate)
        mlp.build()

        neg_count, pos_count = np.bincount(y_train)
        n_samples = len(y_train)
        class_weight = {
            0: (1 / neg_count) * (n_samples / 2.0),
            1: (1 / pos_count) * (n_samples / 2.0),
        }

        mlp.model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[keras_callback],
            verbose=1,
        )
        mlp.save(mlp_path)
        mlflow.keras.log_model(mlp.model, artifact_path="mlp_model")

        # --- Autoencoder ---
        logger.info("Entrenando Autoencoder...")
        X_train_normal = X_train[y_train == 0]

        autoencoder = AutoencoderModel(input_dim=input_dim)
        autoencoder.build()
        autoencoder.model.fit(
            X_train_normal,
            X_train_normal,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[keras_callback],
            verbose=1,
        )
        autoencoder.save(autoencoder_path)
        mlflow.keras.log_model(
            autoencoder.model, artifact_path="autoencoder_model")

        logger.success("Entrenamiento finalizado y registrado en MLflow")


if __name__ == "__main__":
    app()
