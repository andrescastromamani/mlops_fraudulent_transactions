from pathlib import Path
import joblib
import mlflow
import tensorflow as tf

MODEL_NAME = "Fraud_Detection_Model"

# Base de datos en la raíz
DB_PATH = Path("/app/mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")


def load_model():
    """Carga el modelo buscando en MLflow o mediante archivos locales en /app."""
    # 1. Intentar cargar desde MLflow (escaneando todos los experimentos)
    try:
        print(f"[MLflow Config] Buscando experimentos en BD: {DB_PATH}")
        experiments = mlflow.search_experiments()

        for exp in experiments:
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
            )
            if not runs.empty:
                latest_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{latest_run_id}/model"
                print(
                    f"[MLflow SUCCESS] Cargando modelo desde Run ID: {latest_run_id}"
                )
                return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"[MLflow Warning] No se pudo cargar vía MLflow API: {e}")

    # 2. Respaldo Directo: Buscar cualquier archivo de modelo en /app
    found_models = list(Path("/app").rglob("*.keras")) + list(
        Path("/app").rglob("model.pkl")
    )
    for model_path in found_models:
        print(f"[FastAPI SUCCESS] Cargando archivo local desde {model_path}")
        if model_path.suffix == ".keras":
            return tf.keras.models.load_model(model_path, compile=False)
        else:
            return joblib.load(model_path)

    print("[FastAPI ERROR] No se encontraron archivos de modelo en el sistema.")
    return None


def get_model_metadata():
    return {"version": "v1.0.0", "run_id": "Production"}