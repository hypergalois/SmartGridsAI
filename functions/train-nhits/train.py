import mlrun
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NHiTS
import joblib

def train(context, file_path: str, model_name: str = "nhits_model"):
    """
    Entrena un modelo NHiTS y lo loggea en MLRun.
    """

    # Cargar datos
    df = pd.read_parquet(file_path)

    # Agrupación diaria
    df_daily = df.groupby(df['fecha_hora'].dt.date).agg({'consumo_kwh': 'mean'}).reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily['ds'] = pd.to_datetime(df_daily['ds'])

    # Convertir a TimeSeries de Darts
    series = TimeSeries.from_dataframe(df_daily, 'ds', 'y')

    # Entrenar NHiTS
    model = NHiTS(input_chunk_length=30, output_chunk_length=7, n_epochs=100)
    model.fit(series)

    # Guardar modelo
    model_file = "/tmp/nhits_model.pth"
    torch.save(model.model.state_dict(), model_file)

    # Loggear modelo
    context.log_model(model_name,
                      body_path=model_file,
                      model_file=model_file,
                      framework='darts',
                      labels={"algo": "nhits"})

    print(f"✅ Modelo NHiTS entrenado y loggeado.")