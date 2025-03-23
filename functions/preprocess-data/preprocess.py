import mlrun
import pandas as pd

def preprocess_data(context, file_path: str):
    """Preprocesa y registra datos en el Feature Store."""

    # Cargar el dataset
    df = pd.read_csv(file_path)

    # Verificar si las columnas necesarias existen
    required_columns = ['timestamp', 'casa_id', 'consumo_kwh', 'coste_euros']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    # Preprocesamiento de datos
    df['fecha_hora'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['timestamp'], inplace=True)

    df.dropna(inplace=True)
    df = df[(df["consumo_kwh"] > 0) & (df["coste_euros"] > 0)]

    # Obtener el proyecto actual de MLRun
    project = mlrun.get_or_create_project("smartgrids")

    # Crear o cargar el feature set
    feature_set = project.get_feature_set(
        name="consumo_electrico",
        entities=["casa_id", "fecha_hora"],
        description="Consumo eléctrico por casa y tiempo"
    )

    # Ingesta en el feature store
    feature_set.ingest(df)

    # Guardar dataset preprocesado como artifact
    context.log_dataset("consumo_electrico", df=df, format="parquet")

    print("✅ Datos preprocesados y almacenados en el Feature Store.")