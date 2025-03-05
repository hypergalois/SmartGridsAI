import os
os.system("pip install statsmodels scikit-learn")

import mlrun
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

def train_forecasting_model(context, dataset_uri: str):
    """Entrena un modelo de forecasting basado en SARIMA y lo guarda en MLRun."""
    
    # 1️⃣ Cargar el dataset preprocesado desde MLRun
    dataset = mlrun.get_dataitem(dataset_uri)
    df = dataset.as_df()

    # 2️⃣ Convertir la columna de fecha a datetime
    df["fecha"] = pd.to_datetime(df["fecha"])
    df.set_index("fecha", inplace=True)  # Usamos la fecha como índice

    # 3️⃣ Normalizar los datos con StandardScaler
    scaler = StandardScaler()
    df["consumo_kwh"] = scaler.fit_transform(df[["consumo_kwh"]])

    # 4️⃣ División en entrenamiento y validación (80%-20%)
    split_point = int(len(df) * 0.8)
    train, val = df.iloc[:split_point], df.iloc[split_point:]

    # 5️⃣ Entrenar un modelo SARIMA
    model = SARIMAX(train["consumo_kwh"], 
                    order=(2, 1, 2),  # Parámetros ARIMA (p, d, q)
                    seasonal_order=(1, 1, 1, 12),  # Parámetros SARIMA (P, D, Q, s)
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit()

    # 6️⃣ Guardar el modelo en MLRun
    context.log_model("modelo_forecasting_sarima", body=results, model_format="pkl")
    context.log_artifact("scaler", body=scaler, artifact_type="pkl")

    context.logger.info("✅ Modelo SARIMA entrenado y guardado en MLRun")
