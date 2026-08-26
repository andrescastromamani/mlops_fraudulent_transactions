import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List

# Importación corregida a la estructura mlops_fraudulent_transactions
from mlops_fraudulent_transactions.api.model_loader import load_model, get_model_metadata, MODEL_NAME

app = FastAPI(
    title="API de Detección de Fraude en Tarjetas de Crédito",
    description="API MLOps modular con integración a MLflow.",
    version="1.0.0"
)

model = None
model_version_info = {"version": "Desconocida", "run_id": "Desconocido"}

# Carga del Scaler ajustada a la estructura de carpetas actual
BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

scaler = None
if SCALER_PATH.exists():
    scaler = joblib.load(SCALER_PATH)
    print(f"[FastAPI] Scaler cargado exitosamente desde {SCALER_PATH}")
else:
    # Ruta alternativa de respaldo si el modelo está dentro de la raíz
    ALT_SCALER_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "scaler.pkl"
    if ALT_SCALER_PATH.exists():
        scaler = joblib.load(ALT_SCALER_PATH)
        print(f"[FastAPI] Scaler cargado desde ruta alternativa: {ALT_SCALER_PATH}")
    else:
        print(f"[FastAPI WARNING] No se encontró el scaler en {SCALER_PATH}. Aplicando estandarización de respaldo.")

# Medias y desviaciones estándar globales del dataset original para respaldo
MEAN_TIME, STD_TIME = 94813.86, 47488.15
MEAN_AMOUNT, STD_AMOUNT = 88.34, 250.12

def preprocesar_entrada_respaldo(df: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df.copy()
    if "Time" in df_scaled.columns:
        df_scaled["Time"] = (df_scaled["Time"] - MEAN_TIME) / STD_TIME
    if "Amount" in df_scaled.columns:
        df_scaled["Amount"] = (df_scaled["Amount"] - MEAN_AMOUNT) / STD_AMOUNT
    return df_scaled

@app.on_event("startup")
def startup_event():
    global model, model_version_info
    model = load_model()
    model_version_info = get_model_metadata()
    if model:
        print(f"[FastAPI] Modelo v{model_version_info['version']} cargado exitosamente.")
    else:
        print("[FastAPI ERROR] No se pudo cargar el modelo.")

class TransactionFeatures(BaseModel):
    Time: float = Field(..., example=0.0)
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., example=149.62)

class PredictionRequest(BaseModel):
    data: List[TransactionFeatures]

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo no está cargado en memoria."
        )
    
    try:
        input_data = pd.DataFrame([item.dict() for item in payload.data])
        
        # Procesamiento de datos con Scaler pkl o respaldo
        if scaler is not None:
            scaled_data = scaler.transform(input_data)
        else:
            df_preprocessed = preprocesar_entrada_respaldo(input_data)
            scaled_data = df_preprocessed.values

        raw_predictions = model.predict(scaled_data)
        
        results = []
        if hasattr(raw_predictions, "shape") and raw_predictions.shape == scaled_data.shape:
            mse_errors = np.mean(np.square(scaled_data - raw_predictions), axis=1)
        else:
            mse_errors = np.array(raw_predictions).flatten()

        THRESHOLD = 0.50

        for i, mse in enumerate(mse_errors):
            reconstruction_error = float(mse)
            is_fraud = bool(reconstruction_error > THRESHOLD)
            
            prob_fraud = float(1 / (1 + np.exp(-(reconstruction_error - THRESHOLD))))
            prob_fraud = float(np.clip(prob_fraud, 0.0, 1.0))
            prob_legit = float(1.0 - prob_fraud)
            
            confidence = prob_fraud if is_fraud else prob_legit

            results.append({
                "index": i,
                "is_fraud": is_fraud,
                "diagnosis": "Fraude Detectado" if is_fraud else "Transacción Legítima",
                "reconstruction_error": round(reconstruction_error, 4),
                "confidence_score": round(confidence * 100, 2),
                "probabilities_detail": {
                    "fraud": round(prob_fraud, 4),
                    "legitimate": round(prob_legit, 4)
                }
            })

        return {
            "model_metadata": {
                "name": MODEL_NAME,
                "version": model_version_info["version"],
                "run_id": model_version_info["run_id"]
            },
            "total_predictions": len(results),
            "results": results,
            "message": "Inferencia completada con éxito."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")