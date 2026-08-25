import yaml
from pathlib import Path

import numpy as np
from loguru import logger
from pandas import pd
import typer

from mlops_fraudulent_transactions.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR
)
from mlops_fraudulent_transactions.modeling import AutoencoderModel, MLPModel
app = typer.Typer()


@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    train_labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    mlp_path: Path = MODELS_DIR / "mlp_model.h5",
    autoencoder_path: Path = MODELS_DIR / "autoencoder_model.h5",
    params_path: Path = Path("params.yaml"),
) -> None:
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    batch_size = params["train"]["batch_size"]
    learning_rate = params["train"]["learning_rate"]

    logger.info(f"Cargando datos desde {train_features_path}...")
    X_train = pd.read_csv(train_features_path).to_numpy()
    y_train = pd.read_csv(train_labels_path).to_numpy().ravel()
    input_dim = X_train.shape[1]

    # 1. Entrenar MLP
    logger.info("--- Iniciando entrenamiento: MLP Supervisado ---")
    mlp = MLPModel(input_dim=input_dim, learning_rate=learning_rate)
    mlp.build()
    
    neg_count, pos_count = np.bincount(y_train)
    n_samples = len(y_train)
    class_weight = {
        0: (1 / neg_count) * (n_samples / 2.0),
        1: (1 / pos_count) * (n_samples / 2.0),
    }

    mlp.train(
        X_train=X_train,
        y_train=y_train,
        checkpoint_path=mlp_path,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )
    mlp.save(mlp_path)

    # 2. Entrenar Autoencoder
    logger.info("--- Iniciando entrenamiento: Autoencoder de Anomalías ---")
    x_train_normal = X_train[y_train == 0]
    
    autoencoder = AutoencoderModel(input_dim=input_dim)
    autoencoder.build()
    autoencoder.train(
        X_train=x_train_normal,
        checkpoint_path=autoencoder_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    autoencoder.save(autoencoder_path)

    logger.success("✅ Entrenamiento de ambos modelos completado con éxito.")

if __name__ == "__main__":
    app()