import mlrun
import pandas as pd

def preprocess_data(context, file_path: str):
    """Preprocesa el dataset de consumo eléctrico."""
    
    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Convertir la columna fecha a formato datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Filtrar valores negativos en consumo_kwh y coste_euros
    df = df[(df["consumo_kwh"] > 0) & (df["coste_euros"] > 0)]

    # Loggear el dataset preprocesado en MLRun
    context.log_dataset("consumo_electrico_preprocesado", df=df, format="csv")

    context.logger.info(f"✅ Preprocesamiento completado. Filas finales: {len(df)}")
