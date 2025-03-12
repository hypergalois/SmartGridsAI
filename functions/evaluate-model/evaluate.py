import mlrun
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate(context, model_path: str, test_file: str):
    """Evalúa el modelo Prophet."""

    # Cargar el modelo
    model = joblib.load(model_path)

    # Cargar datos de test
    df_test = pd.read_parquet(test_file)
    df_test = df_test.rename(columns={"fecha_hora": "ds", "consumo_kwh": "y"})

    # Predecir
    future = df_test[['ds']]
    forecast = model.predict(future)

    # Métricas
    mse = mean_squared_error(df_test['y'], forecast['yhat'])

    # Loggear las métricas
    context.log_result("mse", mse)
    print(f"✅ Evaluación completa. MSE: {mse}")