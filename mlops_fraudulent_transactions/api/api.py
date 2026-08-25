from mlops_fraudulent_transactions.config import (
    AUTOENCODER_MODEL_PATH,
    MLP_MODEL_PATH,
)
from contextlib import asynccontextmanager
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

# Config the PATH to detect the local package
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


#Global variables to hold the models
mlp_model: keras.Model | None = None
autoencoder_model: keras.Model | None = None
MODEL_INFO = {
    "mlp": str(MLP_MODEL_PATH),
    "autoencoder": str(AUTOENCODER_MODEL_PATH),
}

#Lifecycle event to load models
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlp_model, autoencoder_model
    try:
        if Path(MLP_MODEL_PATH).exists():
            mlp_model = keras.models.load_model(MLP_MODEL_PATH)
            print(f"[FastAPI] MLP cargado exitosamente desde {MLP_MODEL_PATH}")
        else:
            print(
                f"[FastAPI ADVERTENCIA] No existe el archivo: {MLP_MODEL_PATH}")
    except Exception as e:
        print(f"[FastAPI ERROR] No se pudo cargar MLP: {e}")

    try:
        if Path(AUTOENCODER_MODEL_PATH).exists():
            autoencoder_model = keras.models.load_model(AUTOENCODER_MODEL_PATH)
            print(
                f"[FastAPI] Autoencoder cargado exitosamente desde {AUTOENCODER_MODEL_PATH}")
        else:
            print(
                f"[FastAPI ADVERTENCIA] No existe el archivo: {AUTOENCODER_MODEL_PATH}")
    except Exception as e:
        print(f"[FastAPI ERROR] No se pudo cargar Autoencoder: {e}")

    yield

    # Clean up models on shutdown
    mlp_model = None
    autoencoder_model = None


app = FastAPI(
    title="API Fraudulent Transactions MLOps",
    description="API MLOps to predict fraudulent transactions using MLP and Autoencoder models.",
    version="1.0.0",
    lifespan=lifespan,
)

# Eschema definition (Pydantic)
class TransactionFeatures(BaseModel):
    time: float = Field(..., alias="Time")
    v1: float = Field(..., alias="V1")
    v2: float = Field(..., alias="V2")
    v3: float = Field(..., alias="V3")
    v4: float = Field(..., alias="V4")
    v5: float = Field(..., alias="V5")
    v6: float = Field(..., alias="V6")
    v7: float = Field(..., alias="V7")
    v8: float = Field(..., alias="V8")
    v9: float = Field(..., alias="V9")
    v10: float = Field(..., alias="V10")
    v11: float = Field(..., alias="V11")
    v12: float = Field(..., alias="V12")
    v13: float = Field(..., alias="V13")
    v14: float = Field(..., alias="V14")
    v15: float = Field(..., alias="V15")
    v16: float = Field(..., alias="V16")
    v17: float = Field(..., alias="V17")
    v18: float = Field(..., alias="V18")
    v19: float = Field(..., alias="V19")
    v20: float = Field(..., alias="V20")
    v21: float = Field(..., alias="V21")
    v22: float = Field(..., alias="V22")
    v23: float = Field(..., alias="V23")
    v24: float = Field(..., alias="V24")
    v25: float = Field(..., alias="V25")
    v26: float = Field(..., alias="V26")
    v27: float = Field(..., alias="V27")
    v28: float = Field(..., alias="V28")
    amount: float = Field(..., alias="Amount")

    model_config = {
        "populate_by_name": True  # 'Time' o 'time'
    }


class PredictionRequest(BaseModel):
    data: list[TransactionFeatures]

# Endpoints 
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "models": {
            "mlp_loaded": mlp_model is not None,
            "autoencoder_loaded": autoencoder_model is not None,
        },
        "model_paths": MODEL_INFO,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if (mlp_model is not None or autoencoder_model is not None) else "degraded",
        "mlp_loaded": mlp_model is not None,
        "autoencoder_loaded": autoencoder_model is not None,
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if mlp_model is None and autoencoder_model is None:
        raise HTTPException(
            status_code=500,
            detail="Models are not loaded. Please check the server logs for errors during model loading.",
        )

    try:
        # Convert received data to a Pandas DataFrame
        input_data = pd.DataFrame(
            [item.model_dump(by_alias=True) for item in payload.data])
        X = input_data.to_numpy()

        results = []

        mlp_probs = mlp_model.predict(
            X, verbose=0).ravel() if mlp_model is not None else None
        reconstructed = autoencoder_model.predict(
            X, verbose=0) if autoencoder_model is not None else None

        for i in range(len(input_data)):
            result: dict = {"index": i}

            if mlp_probs is not None:
                prob = float(mlp_probs[i])
                result["mlp_fraud_probability"] = round(prob, 4)
                result["mlp_prediction"] = "Fraude" if prob >= 0.5 else "Legítimo"

            if reconstructed is not None:
                mse = float(np.mean(np.square(X[i] - reconstructed[i])))
                result["autoencoder_reconstruction_error"] = round(mse, 6)

            results.append(result)

        return {
            "total_predictions": len(results),
            "results": results,
            "message": "Completed successfully",
        }

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during inference: {str(e)}")
