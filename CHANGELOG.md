# Changelog - Correciones Overfitting y API

## Resumen

Este documento describe los problemas encontrados en el proyecto `mlops_fraudulent_transactions` y las correcciones implementadas para resolver overfitting, fallos en el Autoencoder y problemas en la API de inferencia.

---

## 1. Problema: Overfitting en Modelos

### Diagnóstico

El script `train.py` llamaba directamente a `model.fit()` sin usar los callbacks `EarlyStopping` y `ModelCheckpoint` que estaban definidos en las clases `MLPModel` y `AutoencoderModel`.

**Consecuencia:** Los modelos entrenaban todas las épocas sin parada temprana, lo que provocaba sobreajuste (overfitting).

### Cambios Realizados

#### Archivo: `mlops_fraudulent_transactions/modeling/train.py`

**Líneas 64-87** - Se reemplazó `model.fit()` por `model.train()`:

```python
# ANTES (sin EarlyStopping)
mlp.model.fit(
    X_train,
    y_train,
    validation_split=0.15,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,
    callbacks=[keras_callback],  # Solo log de MLflow
)

# DESPUÉS (con EarlyStopping y ModelCheckpoint)
mlp.train(
    X_train,
    y_train,
    checkpoint_path=mlp_path,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.15,
    class_weight=class_weight,
)
```

**¿Por qué?** El método `train()` de las clases incluye:
- `EarlyStopping(monitor="val_pr_auc", patience=5)` - Detiene el entrenamiento si no hay mejora
- `ModelCheckpoint(save_best_only=True)` - Guarda solo el mejor modelo

---

## 2. Problema: Autoencoder no Detectaba Fraude

### Diagnóstico

El Autoencoder usaba un threshold fijo de `0.5` para clasificar fraudes, pero los errores de reconstrucción no están en el rango `[0, 1]`. Esto causaba que **ninguna transacción** fuera clasificada como fraude (Precision=0, Recall=0, F1=0).

### Cambios Realizados

#### Archivo: `mlops_fraudulent_transactions/modeling/predict.py`

**Línea 34** - Se mantiene el threshold por defecto pero se calcula dinámicamente:

```python
THRESHOLD = 0.5  # Solo para MLP
```

**Líneas 103-125** - Función `run_inference()` modificada:

```python
# ANTES
def run_inference(x_test, mlp_path, autoencoder_path):
    return {
        "MLP_Supervisado": mlp_wrapper.predict(x_test).ravel(),
        "Autoencoder": autoencoder_wrapper.reconstruction_error(x_test),
    }

# DESPUÉS
def run_inference(x_test, x_train_normal, mlp_path, autoencoder_path):
    # Calcular threshold óptimo para Autoencoder usando transacciones normales
    autoencoder_threshold = autoencoder_wrapper.anomaly_threshold(x_train_normal)
    
    return {
        "MLP_Supervisado": mlp_wrapper.predict(x_test).ravel(),
        "Autoencoder": autoencoder_wrapper.reconstruction_error(x_test),
    }, autoencoder_threshold
```

**Líneas 143-155** - Se cargan datos de entrenamiento para calcular threshold:

```python
# Cargar datos de entrenamiento para calcular threshold del Autoencoder
x_train = pd.read_csv(train_features_path).to_numpy()
y_train = pd.read_csv(train_labels_path).to_numpy().ravel()
x_train_normal = x_train[y_train == 0]  # Solo transacciones normales

predictions, autoencoder_threshold = run_inference(...)
summary = evaluator.evaluate(predictions, thresholds={"Autoencoder": autoencoder_threshold})
```

**¿Por qué?** El método `anomaly_threshold()` calcula el percentil 95 de los errores de reconstrucción de transacciones normales, estableciendo un umbral estadísticamente válido.

---

## 3. Problema: Arquitecturas Desalineadas (Notebook vs Producción)

### Diagnóstico

El notebook usaba arquitecturas diferentes a las de los scripts de producción:

| Componente | Notebook | Producción |
|------------|----------|------------|
| MLP capas | 128→64→32 | 64→32 |
| Autoencoder | 16→8→16 | 20→14→20 |

### Cambios Realizados

#### Archivo: `mlops_fraudulent_transactions/modeling/mlp.py`

**Líneas 25-40** - Arquitectura del MLP:

```python
# ANTES
layers.Dense(64, activation="relu"),
layers.BatchNormalization(),
layers.Dropout(self.dropout_rate),
layers.Dense(32, activation="relu"),

# DESPUÉS
layers.Dense(128, activation="relu"),
layers.BatchNormalization(),
layers.Dropout(self.dropout_rate),
layers.Dense(64, activation="relu"),
layers.BatchNormalization(),
layers.Dropout(self.dropout_rate),
layers.Dense(32, activation="relu"),
layers.Dropout(self.dropout_rate),
```

#### Archivo: `mlops_fraudulent_transactions/modeling/autoencoder.py`

**Línea 15** - Encoding dimension:

```python
# ANTES
def __init__(self, input_dim: int, encoding_dim: int = 14) -> None:

# DESPUÉS
def __init__(self, input_dim: int, encoding_dim: int = 8) -> None:
```

**Líneas 19-24** - Arquitectura del Autoencoder:

```python
# ANTES
encoded = layers.Dense(20, activation="tanh")(input_layer)
encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)
decoded = layers.Dense(20, activation="tanh")(encoded)

# DESPUÉS
encoded = layers.Dense(16, activation="relu")(input_layer)
encoded = layers.Dense(self.encoding_dim, activation="relu")(encoded)
decoded = layers.Dense(16, activation="relu")(encoded)
```

**¿Por qué?** Mantener consistencia entre el notebook (prototipado) y la producción (deploy).

---

## 4. Problema: API sin Escalado de Features

### Diagnóstico

La API recibía los datos crudos (`Time`, `Amount`) sin aplicar el mismo `RobustScaler` que se usó durante el entrenamiento. Esto generaba predicciones incorrectas.

### Cambios Realizados

#### Archivo: `mlops_fraudulent_transactions/config.py`

**Líneas 23-24** - Nuevos paths para scalers:

```python
AMOUNT_SCALER_PATH = PROCESSED_DATA_DIR / "amount_scaler.pkl"
TIME_SCALER_PATH = PROCESSED_DATA_DIR / "time_scaler.pkl"
```

#### Archivo: `mlops_fraudulent_transactions/features.py`

**Línea 3** - Agregar import:

```python
import joblib
```

**Líneas 52-55** - Nuevo método `save_scalers()`:

```python
def save_scalers(self, output_dir: Path) -> None:
    """Save fitted scalers for inference."""
    joblib.dump(self._amount_scaler, output_dir / "amount_scaler.pkl")
    joblib.dump(self._time_scaler, output_dir / "time_scaler.pkl")
    logger.info(f"Scalers saved to {output_dir}")
```

**Líneas 103-104** - Llamada en `main()`:

```python
# 5. Guardar scalers para inferencia
engineer.save_scalers(output_dir)
```

#### Archivo: `mlops_fraudulent_transactions/api/api.py`

**Líneas 1-7** - Nuevos imports:

```python
from mlops_fraudulent_transactions.config import (
    AUTOENCODER_MODEL_PATH,
    MLP_MODEL_PATH,
    AMOUNT_SCALER_PATH,
    TIME_SCALER_PATH,
    PROCESSED_DATA_DIR,
)
```

**Líneas 28-30** - Variables globales para scalers:

```python
amount_scaler = None
time_scaler = None
autoencoder_threshold: float = 0.5
```

**Líneas 61-87** - Carga de scalers y cálculo de threshold en `lifespan()`:

```python
# Cargar scalers
try:
    if Path(AMOUNT_SCALER_PATH).exists():
        amount_scaler = joblib.load(AMOUNT_SCALER_PATH)
    if Path(TIME_SCALER_PATH).exists():
        time_scaler = joblib.load(TIME_SCALER_PATH)
except Exception as e:
    print(f"[FastAPI ERROR] No se pudieron cargar scalers: {e}")

# Calcular threshold del Autoencoder
try:
    train_features_path = PROCESSED_DATA_DIR / "train_features.csv"
    train_labels_path = PROCESSED_DATA_DIR / "train_labels.csv"
    if train_features_path.exists() and train_labels_path.exists():
        x_train = pd.read_csv(train_features_path).to_numpy()
        y_train = pd.read_csv(train_labels_path).to_numpy().ravel()
        x_train_normal = x_train[y_train == 0]
        
        from mlops_fraudulent_transactions.modeling import AutoencoderModel
        autoencoder_wrapper = AutoencoderModel(x_train.shape[1])
        autoencoder_wrapper.model = autoencoder_model
        autoencoder_threshold = autoencoder_wrapper.anomaly_threshold(x_train_normal)
except Exception as e:
    print(f"[FastAPI ERROR] No se pudo calcular threshold: {e}")
```

**Líneas 181-193** - Escalado en endpoint `predict()`:

```python
# Escalar Time y Amount
if amount_scaler is not None and time_scaler is not None:
    input_data["scaled_amount"] = amount_scaler.transform(
        input_data["Amount"].values.reshape(-1, 1))
    input_data["scaled_time"] = time_scaler.transform(
        input_data["Time"].values.reshape(-1, 1))
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    X = input_data[feature_cols].to_numpy()
else:
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = input_data[feature_cols].to_numpy()
```

**Línea 213** - Predicción binaria del Autoencoder:

```python
# ANTES
result["autoencoder_reconstruction_error"] = round(mse, 6)

# DESPUÉS
result["autoencoder_reconstruction_error"] = round(mse, 6)
result["autoencoder_prediction"] = "Fraude" if mse > autoencoder_threshold else "Legítimo"
```

**¿Por qué?** Los modelos fueron entrenados con features escalados; la API debe aplicar la misma transformación.

---

## 5. Configuración DVC

#### Archivo: `dvc.yaml`

**Líneas 52-60** - Agregar dependencias al stage `evaluate`:

```yaml
evaluate:
    cmd: python mlops_fraudulent_transactions/modeling/predict.py
    deps:
      - mlops_fraudulent_transactions/modeling/predict.py
      - data/processed/test_features.csv
      - data/processed/test_labels.csv
      - data/processed/train_features.csv    # NUEVO
      - data/processed/train_labels.csv      # NUEVO
      - models/model_mlp.keras
      - models/model_autoencoder.keras
```

**¿Por qué?** `predict.py` ahora necesita los datos de entrenamiento para calcular el threshold del Autoencoder.

---

## Archivos Modificados (Resumen)

| Archivo | Tipo de Cambio |
|---------|----------------|
| `config.py` | Agregar paths de scalers |
| `features.py` | Guardar scalers con joblib |
| `modeling/mlp.py` | Arquitectura 128→64→32 |
| `modeling/autoencoder.py` | Arquitectura 16→8→16 |
| `modeling/train.py` | Usar métodos `train()` con callbacks |
| `modeling/predict.py` | Threshold dinámico para Autoencoder |
| `api/api.py` | Escalado, carga de scalers, predicción binaria |
| `dvc.yaml` | Agregar dependencias |

---

## Cómo Aplicar los Cambios

```bash
# 1. Re-entrenar modelos con nuevos parámetros
dvc repro

# 2. Verificar métricas
dvc metrics show

# 3. Reiniciar la API
uvicorn mlops_fraudulent_transactions.api.api:app --reload
```

---

## Resultados Esperados

| Métrica | Antes | Después (estimado) |
|---------|-------|-------------------|
| MLP Precision | 0.102 | > 0.15 |
| MLP F1-Score | 0.184 | > 0.25 |
| Autoencoder F1 | 0.000 | > 0.10 |
| Autoencoder Recall | 0.000 | > 0.50 |
