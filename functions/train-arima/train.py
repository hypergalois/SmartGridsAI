import mlrun
import pandas as pd
import pmdarima as pm
import joblib

def train(context, file_path: str):
    """
    Entrena un modelo ARIMA y lo loggea en MLRun.
    """

    # Cargar datos
    df = pd.read_parquet(file_path)

    # Agrupación diaria
    df_daily = df.groupby(df['fecha_hora'].dt.date).agg({'consumo_kwh': 'mean'}).reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily['ds'] = pd.to_datetime(df_daily['ds'])

    # Crear modelo ARIMA
    model = pm.auto_arima(df_daily['y'], seasonal=True, m=7,
                          suppress_warnings=True, stepwise=True)

    # Guardar modelo
    model_file = "/tmp/arima_model.pkl"
    joblib.dump(model, model_file)

    # Loggear modelo
    context.log_model("arima_model",
                      body_path=model_file,
                      model_file=model_file,
                      framework='pmdarima',
                      labels={"algo": "arima"})

    print(f"✅ Modelo ARIMA entrenado y loggeado.")