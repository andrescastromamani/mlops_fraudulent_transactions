from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mlops_fraudulent_transactions.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()


class CreditCardDataset:
    """Loads the credit card transactions dataset from disk and exports it."""

    def __init__(self, data_path: Path = RAW_DATA_DIR / "creditcard.csv") -> None:
        self.data_path = data_path
        self.frame: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Read the CSV file and keep it in the frame attribute."""
        logger.info(f"Loading dataset from {self.data_path}")
        self.frame = pd.read_csv(self.data_path)
        logger.success(f"Dataset loaded with shape {self.frame.shape}")
        return self.frame

    def save(self, output_path: Path) -> None:
        """Save the loaded frame to a specified output destination."""
        if self.frame is None:
            raise RuntimeError("Call load() before saving the dataset.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(output_path, index=False)
        logger.success(f"Dataset saved to {output_path}")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "creditcard.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
) -> None:
    dataset = CreditCardDataset(data_path=input_path)
    dataset.load()
    dataset.save(output_path=output_path)


if __name__ == "__main__":
    app()
