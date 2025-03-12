import mlrun
from storey import MapClass
import pandas as pd

def preprocess_data(context, file_path: str):
    """Preprocesa y registra datos en el Feature Store."""
    
    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Eliminar columnas originales de fecha y hora
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Filtrar valores negativos en consumo_kwh y coste_euros
    df = df[(df["consumo_kwh"] > 0) & (df["coste_euros"] > 0)]

    # Obtener el proyecto de MLRun
    project = mlrun.get_or_create_project("smartgrids")

    # Definir el Feature Store
    feature_set = project.get_feature_set(
        name="consumo_electrico",
        entities=["casa_id", "fecha_hora"],  # Llaves primarias
        description="Consumo eléctrico por casa y tiempo"
    )

    # Ingerir los datos en el Feature Store
    feature_set.ingest(df)

    # Guardar dataset preprocesado en Parquet
    context.log_dataset("consumo_electrico", df=df, format="parquet")

    print("✅ Datos preprocesados y almacenados en el Feature Store.")
