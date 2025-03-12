import mlrun
import pandas as pd
from prophet import Prophet
import joblib

def train(context, file_path: str, model_name: str = "prophet_model"):
    """Entrena un modelo Prophet y lo guarda como artifact."""

    # Cargar los datos
    df = pd.read_parquet(file_path)
    
    # Formatear para Prophet
    df_prophet = df.rename(columns={"fecha_hora": "ds", "consumo_kwh": "y"})

    # Entrenar modelo
    model = Prophet()
    model.fit(df_prophet)

    # Guardar el modelo
    model_file = f"/tmp/{model_name}.pkl"
    joblib.dump(model, model_file)

    # Loggear el modelo como artifact en MLRun
    context.log_model(model_name,
                      body_path=model_file,
                      model_file=model_file,
                      framework='prophet',
                      labels={"algo": "prophet"})

    print(f"✅ Modelo {model_name} entrenado y guardado.")