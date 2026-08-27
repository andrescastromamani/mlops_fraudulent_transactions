from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from mlops_fraudulent_transactions.config import PROJ_ROOT

DATA_PROCESSED_PATH = PROJ_ROOT / "data" / "processed" / "dataset.csv"
FIGURES_DIR = PROJ_ROOT / "reports" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PROCESSED_PATH.exists():
        logger.error(f"No se encontró el archivo en {DATA_PROCESSED_PATH}")
        return

    logger.info("Cargando dataset para generar gráficos...")
    df = pd.read_csv(DATA_PROCESSED_PATH)

    # 1. Gráfico de distribución de clases
    plt.figure(figsize=(6, 4))
    if "Class" in df.columns:
        df["Class"].value_counts().plot(kind="bar", color=["navy", "crimson"])
        plt.title("Distribución de Transacciones (0: Normal, 1: Fraude)")
        plt.xlabel("Clase")
        plt.ylabel("Cantidad")
        plt.xticks(rotation=0)

    class_dist_path = FIGURES_DIR / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(class_dist_path)
    plt.close()
    logger.success(f"Gráfico guardado en: {class_dist_path}")

    # 2. Mapa de calor de correlación
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Mapa de Calor de Correlación de Características")

    corr_heatmap_path = FIGURES_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(corr_heatmap_path)
    plt.close()
    logger.success(f"Gráfico guardado en: {corr_heatmap_path}")

    # 3. Distribución de montos por clase
    plt.figure(figsize=(8, 5))
    if "Amount" in df.columns and "Class" in df.columns:
        sns.boxplot(x="Class", y="Amount", data=df, hue="Class", palette=["navy", "crimson"], legend=False)
        plt.yscale("log")
        plt.title("Distribución de Montos por Clase (Escala Logarítmica)")
        plt.xlabel("Clase (0: Normal, 1: Fraude)")
        plt.ylabel("Monto (Log)")

    amounts_path = FIGURES_DIR / "amounts_by_class.png"
    plt.tight_layout()
    plt.savefig(amounts_path)
    plt.close()
    logger.success(f"Gráfico guardado en: {amounts_path}")

    # 4. Distribución de transacciones en el tiempo (Último gráfico)
    plt.figure(figsize=(10, 4))
    if "Time" in df.columns and "Class" in df.columns:
        sns.histplot(data=df, x="Time", hue="Class", element="step", stat="density", common_norm=False, palette=["navy", "crimson"])
        plt.title("Distribución de Transacciones a lo largo del Tiempo")
        plt.xlabel("Tiempo (segundos)")
        plt.ylabel("Densidad")

    time_path = FIGURES_DIR / "time_distribution.png"
    plt.tight_layout()
    plt.savefig(time_path)
    plt.close()
    logger.success(f"Gráfico guardado en: {time_path}")


if __name__ == "__main__":
    main()