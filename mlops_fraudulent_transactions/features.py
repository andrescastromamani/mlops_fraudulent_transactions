from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import typer
import yaml  # <-- Importar PyYAML

from mlops_fraudulent_transactions.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        self._amount_scaler.fit(self.X_train["Amount"].values.reshape(-1, 1))
        self._time_scaler.fit(self.X_train["Time"].values.reshape(-1, 1))
        self.X_train = self._scale(self.X_train)
        self.X_test = self._scale(self.X_test)
        logger.info(f"Train {self.X_train.shape} - Test {self.X_test.shape} after scaling")

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
    input_path: Path = RAW_DATA_DIR / "creditcard.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = Path("params.yaml"),
) -> None:
    # Leer params.yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["split_ratio"]
    random_state = params["prepare"]["random_state"]

    logger.info("Cargando dataset...")
    frame = pd.read_csv(input_path)

    engineer = FeatureEngineer(test_size=test_size, random_state=random_state)
    engineer.prepare(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar features y labels separados para DVC
    engineer.X_train.to_csv(output_dir / "train_features.csv", index=False)
    engineer.y_train.to_csv(output_dir / "train_labels.csv", index=False)
    engineer.X_test.to_csv(output_dir / "test_features.csv", index=False)
    engineer.y_test.to_csv(output_dir / "test_labels.csv", index=False)

    logger.success(f"Features guardadas correctamente en {output_dir}")

if __name__ == "__main__":
    app()