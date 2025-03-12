import os
import subprocess
import io

# Instalar Prophet si no está instalado
try:
    from prophet import Prophet
except ImportError:
    subprocess.check_call(["pip", "install", "prophet"])
    from prophet import Prophet  # Importar después de instalar

import pandas as pd
import matplotlib.pyplot as plt
import holidays
from prophet import Prophet

def train_prophet_model(context):
    """
    Entrena un modelo Prophet utilizando datos desde una ruta S3 y registra los resultados en MLRun.
    """
    # Ruta directa del dataset en S3
    s3_path = "s3://smartgrids-bucket/datasets/guardar-dataset-minio/0/consumo_electrico.csv"

    # Descargar dataset directamente desde S3
    dataset = context.get_dataitem(s3_path)  # MLRun obtiene el archivo S3
    data_bytes = dataset.get()  # Obtener los bytes del archivo
    df = pd.read_csv(io.BytesIO(data_bytes))  # Convertir bytes en CSV legible por Pandas

    # Preprocesamiento
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.dropna()
    df = df[df['consumo_kwh'] > 0]
    df_daily = df.groupby(df['timestamp'].dt.date).agg({'consumo_kwh': 'mean'}).reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily['ds'] = pd.to_datetime(df_daily['ds'])

    # Agregar festivos en España
    es_holidays = holidays.Spain(years=range(df_daily['ds'].dt.year.min(), df_daily['ds'].dt.year.max() + 1))
    holidays_df = pd.DataFrame({'ds': list(es_holidays.keys()), 'holiday': list(es_holidays.values())})

    # Entrenar el modelo Prophet
    model = Prophet(holidays=holidays_df)
    model.fit(df_daily)

    # Predicción
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Guardar el modelo y los resultados en MLRun
    context.log_dataset("forecast", df=forecast, format="parquet")
    context.log_model("prophet_model", body=model, model_format="pkl")

    # Visualización
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_daily['ds'], df_daily['y'], label='Consumo Real')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicción')
    ax.legend()
    ax.set_title("Predicción de Consumo Energético con Prophet")
    plt.savefig("forecast_plot.png")

    # Registrar la imagen en MLRun
    context.log_artifact("forecast_plot", local_path="forecast_plot.png")

    return forecast
