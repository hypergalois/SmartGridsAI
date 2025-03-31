# -*- coding: utf-8 -*-
"""Producción_por_placa.ipynb

El csv por ahora descargarlo manual.

Original file is located at
    https://colab.research.google.com/drive/1C_PTsL1Gj_nxswHwFwIu-vCGiHS_PZgn
"""

!pip install prophet

!unzip dataset_d3_filtrado.zip

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# 1) CARGA DE DATOS
file_path = "dataset_d3_filtrado.csv"
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])

# 2) FILTRADO Y NORMALIZACIÓN
df = df[df['num_placas'] > 0]
df['produccion_por_placa'] = df['produccion_kWh'] / df['num_placas']
df.sort_values(by='datetime', inplace=True)

# 3) AGRUPACIÓN POR CASA Y HORA
df_grouped = (
    df.groupby(['id_casa', 'datetime'], as_index=False)
      .agg({'produccion_por_placa': 'mean'})
)

# 4) MODELADO Y PREDICCIÓN POR CADA CASA
resultados_por_casa = {}

casas_unicas = df_grouped['id_casa'].unique()

for casa_id in casas_unicas:
    df_casa = df_grouped[df_grouped['id_casa'] == casa_id].copy()
    df_casa.rename(columns={'datetime': 'ds', 'produccion_por_placa': 'y'}, inplace=True)

    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
        interval_width=0.90
    )

    try:
        modelo.fit(df_casa)
        future = modelo.make_future_dataframe(periods=48, freq='H')
        forecast = modelo.predict(future)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

        resultados_por_casa[casa_id] = {
            'modelo': modelo,
            'forecast': forecast,
            'datos_entrenamiento': df_casa
        }

        if casa_id == casas_unicas[0]:
            fig = modelo.plot(forecast)
            plt.title(f"Predicción producción por placa - Casa {casa_id}")
            plt.xlabel("Fecha/Hora")
            plt.ylabel("Producción (kWh) por placa")
            plt.grid(True)
            plt.show()

    except Exception as e:
        print(f"Error al procesar casa {casa_id}: {e}")

# 5) IMPRIMIR ALGUNOS RESULTADOS
print("\nPredicciones recientes por casa:")
for casa_id, datos in resultados_por_casa.items():
    print(f"\nCasa {casa_id}:")
    print(datos['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

import matplotlib.pyplot as plt

# 1) PRODUCCIÓN MEDIA PREDICHA POR CASA
media_predicha = {
    casa_id: datos['forecast']['yhat'].mean()
    for casa_id, datos in resultados_por_casa.items()
}

plt.figure(figsize=(12, 6))
plt.bar(media_predicha.keys(), media_predicha.values())
plt.title("📈 Producción media predicha por placa - por casa")
plt.xlabel("ID de casa")
plt.ylabel("Producción media (kWh)")
plt.grid(True)
plt.show()

# 2) VARIABILIDAD DE PRODUCCIÓN (DESVIACIÓN ESTÁNDAR)
variabilidad_predicha = {
    casa_id: datos['forecast']['yhat'].std()
    for casa_id, datos in resultados_por_casa.items()
}

plt.figure(figsize=(12, 6))
plt.bar(variabilidad_predicha.keys(), variabilidad_predicha.values(), color='orange')
plt.title("📊 Variabilidad (STD) de la producción predicha - por casa")
plt.xlabel("ID de casa")
plt.ylabel("Desviación estándar (kWh)")
plt.grid(True)
plt.show()

# 3) TOP 5 CASAS CON MENOR PRODUCCIÓN PROMEDIO
casas_menor_prod = sorted(media_predicha.items(), key=lambda x: x[1])[:5]
print("\n🏠 Casas con menor producción promedio esperada:")
for casa_id, promedio in casas_menor_prod:
    print(f"Casa {casa_id}: {promedio:.3f} kWh")

# 4) CURVAS SUPERPUESTAS DE PRODUCCIÓN DE ALGUNAS CASAS
casas_a_comparar = [item[0] for item in casas_menor_prod]  # Por ejemplo, las 5 con menor producción

plt.figure(figsize=(14, 6))
for casa_id in casas_a_comparar:
    forecast = resultados_por_casa[casa_id]['forecast']
    plt.plot(forecast['ds'], forecast['yhat'], label=f'Casa {casa_id}')

plt.title("📆 Producción predicha (yhat) para casas con menor rendimiento")
plt.xlabel("Fecha/Hora")
plt.ylabel("Producción (kWh)")
plt.legend()
plt.grid(True)
plt.show()