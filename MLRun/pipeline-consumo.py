import pandas as pd
import numpy as np
import xgboost as xgb
import mlrun
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlrun.artifacts import PlotArtifact
from io import BytesIO

@mlrun.handler()
def entrenamiento_consumo_xgboost(context, dataset_path: str):
    # Cargar los datos
    df = pd.read_csv(dataset_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.sort_values(['casa_id', 'timestamp'], inplace=True)

    # Crear features temporales
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Crear lag features
    df['lag_1'] = df.groupby('casa_id')['consumo_kwh'].shift(1)
    df['lag_24'] = df.groupby('casa_id')['consumo_kwh'].shift(24)
    df = df.dropna(subset=['lag_1', 'lag_24'])

    df.sort_values('timestamp', inplace=True)
    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]

    features = ['hour', 'day', 'month', 'dayofweek', 'lag_1', 'lag_24']
    target = 'consumo_kwh'

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    # Definición del modelo y grid search
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5],
        'learning_rate': [0.1],
        'subsample': [0.8]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv,
                               scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    context.log_result("MAE", mae)
    context.log_result("RMSE", rmse)

    # Guardar el modelo entrenado
    context.log_model("modelo_xgboost_consumo",
                      body=best_xgb,
                      model_file="model.pkl",
                      model_format="pkl")

    # Gráfico de predicción vs real
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_test['timestamp'], y_test, label='Real')
    ax.plot(df_test['timestamp'], y_pred, label='Predicción')
    ax.set_title("Consumo Eléctrico: Real vs Predicción")
    ax.legend()
    plt.xticks(rotation=45)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    context.log_artifact(PlotArtifact("grafico_prediccion_vs_real", body=buf, format="png"))

    context.logger.info("Entrenamiento XGBoost completado y artefactos registrados.")
