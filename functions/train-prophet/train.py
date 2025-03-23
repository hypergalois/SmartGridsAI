import mlrun
import pandas as pd
from prophet import Prophet
import joblib
import holidays
from prophet.diagnostics import cross_validation, performance_metrics

def train(context, file_path: str, country_holidays: str = 'Spain'):
    """
    Entrena un modelo Prophet, hace tuning de hiperparámetros y loggea el modelo.
    """

    # Cargar datos
    df = pd.read_parquet(file_path)

    # Agrupación diaria (puedes cambiarlo según tus datos)
    df_daily = df.groupby(df['fecha_hora'].dt.date).agg({'consumo_kwh': 'mean'}).reset_index()
    df_daily.columns = ['ds', 'y']
    df_daily['ds'] = pd.to_datetime(df_daily['ds'])

    # Agregar festivos
    years = range(df_daily['ds'].dt.year.min(), df_daily['ds'].dt.year.max()+1)
    es_holidays = holidays.CountryHoliday(country_holidays, years=years)
    holidays_df = pd.DataFrame({'ds': list(es_holidays.keys()), 'holiday': 'public_holiday'})

    # Hyperparameter grid
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.1, 1.0, 10.0]
    }

    best_rmse = float('inf')
    best_model = None

    # Búsqueda en rejilla
    from sklearn.model_selection import ParameterGrid
    for params in ParameterGrid(param_grid):
        model = Prophet(**params)
        model.add_country_holidays(country_name=country_holidays)
        model.fit(df_daily)

        df_cv = cross_validation(model, initial='365 days', period='30 days', horizon='7 days')
        df_p = performance_metrics(df_cv)
        rmse = df_p['rmse'].mean()

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Guardar modelo
    model_file = "/tmp/prophet_model.pkl"
    joblib.dump(best_model, model_file)

    # Loggear modelo en MLRun
    context.log_model("prophet_model",
                      body_path=model_file,
                      model_file=model_file,
                      framework='prophet',
                      metrics={'rmse': best_rmse},
                      labels={"algo": "prophet"})

    print(f"✅ Modelo Prophet entrenado y loggeado con RMSE: {best_rmse}")