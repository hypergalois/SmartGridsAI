import mlrun
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(context, model_path: str, test_file: str):
    """
    Evalúa cualquier modelo sklearn/prophet/arima
    """

    # Cargar modelo
    model = joblib.load(model_path)

    # Cargar test
    df_test = pd.read_parquet(test_file)
    df_test = df_test.groupby(df_test['fecha_hora'].dt.date).agg({'consumo_kwh': 'mean'}).reset_index()
    df_test.columns = ['ds', 'y']
    df_test['ds'] = pd.to_datetime(df_test['ds'])

    # Prophet / ARIMA / sklearn predicciones
    try:
        if hasattr(model, 'predict'):
            forecast = model.predict(df_test['ds'])
        else:
            forecast = model.predict(n_periods=len(df_test))
    except Exception as e:
        raise ValueError(f"Error al predecir: {str(e)}")

    mse = mean_squared_error(df_test['y'], forecast)
    mae = mean_absolute_error(df_test['y'], forecast)

    context.log_result("mse", mse)
    context.log_result("mae", mae)

    print(f"✅ Evaluación completa. MSE: {mse}, MAE: {mae}")