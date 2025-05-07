# ────────────────────────────────────────────────────────────────────────
# prediccion_produccion/app.py
# FastAPI + Prophet: API de predicción de producción y consumo,
# y servidor de archivos estáticos
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import json
import pathlib

# ─── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent            # …/prediccion_produccion
WEB  = ROOT.parent / "web"                      # …/web
MODEL_PROD_PATH   = ROOT / "models/prophet_best_5558.pkl"
PARAMS_PROD_PATH  = ROOT / "models/prophet_best_5558_params.json"
MODEL_CONS_PATH   = ROOT / "models/modelo_prophet_consumo.pkl"

# ─── FastAPI & CORS ─────────────────────────────────────────────────────
app = FastAPI(title="EcoTrack API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← restringe en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Esquemas de entrada/salida ────────────────────────────────────────
class PredictItem(BaseModel):
    ds: str
    irradiancia_W_m2: float
    temperatura: float
    nubosidad: float
    lag1: float
    lag7: float

class PredictOut(BaseModel):
    ds: str
    prod_placa: float

class ConsumoItem(BaseModel):
    ds: str
    temperatura: float
    coste_euros: float
    velmedia: float
    sol: float
    racha: float
    prec_log: float
    y_lag_1: float
    y_lag_2: float
    y_lag_24: float
    y_lag_48: float
    y_lag_72: float
    y_lag_168: float
    coste_euros_lag_1: float
    coste_euros_lag_2: float
    coste_euros_lag_24: float
    coste_euros_lag_48: float
    y_moving_avg_3: float
    y_moving_avg_6: float
    y_moving_avg_24: float
    week_avg: float
    trend_diff: float
    coste_euros_moving_avg_3: float
    coste_euros_moving_avg_6: float
    coste_euros_moving_avg_24: float

class ConsumoOut(BaseModel):
    ds: str
    cons_kwh: float

# ─── Carga de modelos ───────────────────────────────────────────────────
prod_model = joblib.load(MODEL_PROD_PATH)
with open(PARAMS_PROD_PATH) as f:
    prod_params = json.load(f)

cons_model = joblib.load(MODEL_CONS_PATH)

# ─── POST /predict (producción) ────────────────────────────────────────
@app.post("/api/predict", response_model=List[PredictOut])
async def predict_produccion(items: List[PredictItem]):
    df = pd.DataFrame([i.dict() for i in items])
    df["ds"] = pd.to_datetime(df["ds"])
    try:
        fcst = prod_model.predict(df)
    except Exception as e:
        raise HTTPException(500, f"Error en predict producción: {e}")
    out = []
    for i, row in df.iterrows():
        y = fcst.loc[i]
        prod = float(np.expm1(y.yhat))
        # no puede ser negativo
        prod = max(prod, 0.0)
        out.append(PredictOut(
                ds = row["ds"].strftime("%Y-%m-%d %H:%M:%S"),
            prod_placa = prod,
        ))
    return out

# ─── POST /predict-consumo (consumo) ────────────────────────────────────
@app.post("/api/predict-consumo", response_model=List[ConsumoOut])
async def predict_consumo(items: List[ConsumoItem]):
    df = pd.DataFrame([i.dict() for i in items])
    df["ds"] = pd.to_datetime(df["ds"])
    try:
        fcst = cons_model.predict(df)  # Prophet con regresores
    except Exception as e:
        raise HTTPException(500, f"Error en predict consumo: {e}")
    out = []
    for i, row in df.iterrows():
        y = fcst.loc[i]
        out.append(ConsumoOut(
            ds       = row["ds"].strftime("%Y-%m-%d %H:%M:%S"),
            cons_kwh = float(y.yhat),
        ))
    return out

# ─── Archivos estáticos (HTML, CSS, JS…) ──────────────────────────────
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
