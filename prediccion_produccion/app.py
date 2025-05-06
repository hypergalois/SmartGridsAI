# ────────────────────────────────────────────────────────────────────────
# prediccion_produccion/app.py
# FastAPI + Prophet: API de predicción y servidor de archivos estáticos
# ────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pandas as pd, numpy as np, joblib, json, pathlib

# ─── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent            # …/prediccion_produccion
WEB = ROOT.parent / "web"

MODEL_PATH  = ROOT / "models/prophet_best_5558.pkl"
PARAMS_PATH = ROOT / "models/prophet_best_5558_params.json"

# ─── FastAPI & CORS ─────────────────────────────────────────────────────
app = FastAPI(title="EcoTrackAPI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← restringe en producción
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
    yhat: float
    yhat_lower: float
    yhat_upper: float
    prod_placa: float

# ─── Carga del modelo ───────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
with open(PARAMS_PATH) as f:
    best_params = json.load(f)

# ─── POST /predict ──────────────────────────────────────────────────────
@app.post("/predict", response_model=List[PredictOut])
async def predict(items: List[PredictItem]):
    """
    Devuelve la predicción de producción (kWh) para cada fila recibida.
    """
    # 1. dataframe
    df = pd.DataFrame([i.dict() for i in items])
    df["ds"] = pd.to_datetime(df["ds"])

    # 2. predicción
    try:
        fcst = model.predict(df)
    except Exception as e:
        raise HTTPException(500, f"Error en predict(): {e}")

    # 3. formateo de salida
    out: List[PredictOut] = []
    for i, row in df.iterrows():
        y = fcst.loc[i]
        out.append(PredictOut(
            ds          = row["ds"].strftime("%Y-%m-%d"),
            yhat        = float(y.yhat),
            yhat_lower  = float(y.yhat_lower),
            yhat_upper  = float(y.yhat_upper),
            prod_placa  = float(np.expm1(y.yhat)),
        ))
    return out

# ─── Archivos estáticos (HTML, CSS, JS, imgs…) ─────────────────────────
#  * html=True hace que si pides "/" entregue index.html automáticamente
#  * Cualquier otro path (p.ej. /stats.html) busca el archivo del mismo
#    nombre dentro de la carpeta WEB. Si no existe → 404 clásico.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
