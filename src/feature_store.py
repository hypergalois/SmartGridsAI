import mlrun
import pandas as pd
from mlrun.feature_store import FeatureSet
from mlrun.datastore import DataItem

def store_features(context, dataset_uri: str):
    """
    Carga los datos desde MinIO (.csv) y los almacena en el Feature Store de MLRun.
    """

    # 📌 Convertir la URI del dataset en un objeto DataItem de MLRun
    dataset = mlrun.get_dataitem(dataset_uri)

    # 📌 Cargar el dataset desde MinIO asegurando que el timestamp es datetime
    df = dataset.as_df(parse_dates=["timestamp"])

    # 📌 Definir el Feature Set en MLRun
    feature_set = FeatureSet(
        "consumo_electrico_fs",
        entities=["casa_id"],  # 📌 Identificador único de la entidad
        timestamp_key="timestamp",  # 📌 Clave de tiempo para series temporales
        description="Feature store para datos de consumo eléctrico",
    )

    # 📌 Ingestar datos al Feature Store
    mlrun.feature_store.ingest(
        feature_set, 
        df,
        infer_options=mlrun.feature_store.InferOptions.default(),  # Detectar tipos de datos automáticamente
    )

    # 📌 Registrar el Feature Set en MLRun
    context.log_result("feature_store", "consumo_electrico_fs")

    print("✅ Datos almacenados en el Feature Store correctamente.")

# 🔹 Ejecutar la función asegurando que MLRun la gestione correctamente
if __name__ == "__main__":
    ctx = mlrun.get_or_create_ctx("store_features_consumo")
    dataset_uri = "store://datasets/smartgrids/guardar-dataset-minio_consumo_electrico#0:latest"  # 📌 Ruta del dataset en MinIO (desde MLRun)

    # 📌 Llamar a la función con el DataItem corregido
    store_features(ctx, dataset_uri)
