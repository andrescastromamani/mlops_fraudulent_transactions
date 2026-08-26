from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import typer
import yaml

from mlops_fraudulent_transactions.config import PROCESSED_DATA_DIR

app = typer.Typer()

class FeatureEngineer:
    """Splits the dataset and scales numeric features without data leakage."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self._amount_scaler = RobustScaler()
        self._time_scaler = RobustScaler()
        self.x_train: pd.DataFrame | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def prepare(
        self,
        frame: pd.DataFrame,
        target: str = "Class",
    ) -> None:
        """Split the frame and scale Time/Amount using scalers fit on train only."""
        features = frame.drop(target, axis=1)
        labels = frame[target]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        self._amount_scaler.fit(self.x_train["Amount"].values.reshape(-1, 1))
        self._time_scaler.fit(self.x_train["Time"].values.reshape(-1, 1))
        self.x_train = self._scale(self.x_train)
        self.x_test = self._scale(self.x_test)
        logger.info(f"Train {self.x_train.shape} - Test {self.x_test.shape} after scaling")

    def save_scalers(self, output_dir: Path) -> None:
        """Save fitted scalers for inference."""
        joblib.dump(self._amount_scaler, output_dir / "amount_scaler.pkl")
        joblib.dump(self._time_scaler, output_dir / "time_scaler.pkl")
        logger.info(f"Scalers saved to {output_dir}")

    def class_weights(self) -> dict:
        """Compute per-class weights to counter the class imbalance."""
        if self.y_train is None:
            raise RuntimeError("Call prepare() before computing class weights.")
        neg_count, pos_count = np.bincount(self.y_train)
        n_samples = len(self.y_train)
        return {
            0: (1 / neg_count) * (n_samples / 2.0),
            1: (1 / pos_count) * (n_samples / 2.0),
        }

    def _scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace Time/Amount with their robust-scaled versions."""
        scaled = frame.copy()
        scaled["scaled_amount"] = self._amount_scaler.transform(
            scaled["Amount"].values.reshape(-1, 1)
        )
        scaled["scaled_time"] = self._time_scaler.transform(scaled["Time"].values.reshape(-1, 1))
        return scaled.drop(["Time", "Amount"], axis=1)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = Path("params.yaml"),
) -> None:
    # 1. Leer parámetros desde params.yaml
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["split_ratio"]
    random_state = params["prepare"]["random_state"]

    # 2. Cargar datos preprocesados de dataset.py
    logger.info(f"Cargando dataset desde {input_path}...")
    frame = pd.read_csv(input_path)

    # 3. Aplicar separación y escalado de características
    engineer = FeatureEngineer(test_size=test_size, random_state=random_state)
    engineer.prepare(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Guardar conjuntos divididos para DVC
    engineer.x_train.to_csv(output_dir / "train_features.csv", index=False)
    engineer.y_train.to_csv(output_dir / "train_labels.csv", index=False)
    engineer.x_test.to_csv(output_dir / "test_features.csv", index=False)
    engineer.y_test.to_csv(output_dir / "test_labels.csv", index=False)

    # 5. Guardar scalers para inferencia
    engineer.save_scalers(output_dir)

    logger.success(f"Features guardadas correctamente en {output_dir}")


if __name__ == "__main__":
    app()