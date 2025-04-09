import os
import pandas as pd
import numpy as np
import joblib
import mlrun
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


@mlrun.handler()
def entrenar_modelos_prophet(context, file_path: str = "dataset_d3_filtrado.csv"):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['num_placas'] > 0]
    df['produccion_por_placa'] = df['produccion_kWh'] / df['num_placas']
    df['nubosidad'] = df['nubosidad'].fillna(0)
    df.sort_values(by='datetime', inplace=True)
    df['produccion_dia_anterior'] = df.groupby('id_casa')['produccion_por_placa'].shift(24).fillna(0)

    df_grouped = df.groupby(['id_casa', 'datetime', 'nubosidad', 'festivo', 'produccion_dia_anterior'], as_index=False)                   .agg({'produccion_por_placa': 'mean'})

    os.makedirs("modelos_prophet", exist_ok=True)
    os.makedirs("resultados_cross_validation", exist_ok=True)

    resultados_por_casa = []
    errores = []

    ids_casas = df_grouped['id_casa'].unique()

    for casa_id in ids_casas:
        df_casa = df_grouped[df_grouped['id_casa'] == casa_id].copy()
        df_prophet = df_casa.rename(columns={'datetime': 'ds', 'produccion_por_placa': 'y', 'festivo': 'holiday'})

        festivos = df_prophet[df_prophet['holiday'] == 1][['ds']].copy()
        festivos['holiday'] = 'festivo'

        modelo = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10,
            interval_width=0.90,
            holidays=festivos
        )

        modelo.add_regressor('nubosidad')
        modelo.add_regressor('produccion_dia_anterior')

        try:
            modelo.fit(df_prophet)
            modelo_path = f"modelos_prophet/modelo_casa_{casa_id}.pkl"
            joblib.dump(modelo, modelo_path)
            context.log_model(f"modelo_prophet_casa_{casa_id}", model_file=modelo_path)

            df_cv = cross_validation(
                modelo,
                initial='2160 hours',
                period='480 hours',
                horizon='48 hours',
                parallel='processes'
            )

            df_metrics = performance_metrics(df_cv)
            mae = df_metrics['mae'].mean()
            rmse = df_metrics['rmse'].mean()

            context.log_result(f"mae_casa_{casa_id}", mae)
            context.log_result(f"rmse_casa_{casa_id}", rmse)

            forecast = modelo.predict(df_prophet)

            plt.figure(figsize=(12, 6))
            plt.plot(df_prophet['ds'], df_prophet['y'], label='Real')
            plt.plot(forecast['ds'], forecast['yhat'], label='Predicción')
            plt.title(f'Predicción de Producción - Casa {casa_id}')
            plt.legend()
            fig_path = f"pred_prophet_casa_{casa_id}.png"
            plt.savefig(fig_path)
            context.log_artifact(f"grafica_casa_{casa_id}", src_path=fig_path)

            resultados_por_casa.append({
                'id_casa': casa_id,
                'mae': mae,
                'rmse': rmse
            })

        except Exception as e:
            context.logger.warn(f"Error en casa {casa_id}: {e}")
            errores.append((casa_id, str(e)))
            continue

    resumen_path = "resumen_metricas_prophet.csv"
    pd.DataFrame(resultados_por_casa).to_csv(resumen_path, index=False)
    context.log_artifact("resumen_metricas", src_path=resumen_path)
