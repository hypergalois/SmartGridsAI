import pandas as pd
import mlrun

def preprocess(context):
    """
    Función de preprocesamiento de datos para el modelo de predicción eléctrica.
    """

    # 📌 URI del dataset en MinIO (MLRun Feature Store)
    dataset_uri = "store://datasets/smartgrids/guardar-dataset-minio_consumo_electrico#0:latest"

    # 📌 Cargar datos desde MinIO usando MLRun
    dataset = mlrun.get_dataitem(dataset_uri)
    df = dataset.as_df(parse_dates=["timestamp"])  # Asegurar conversión correcta del timestamp

    # 📌 Convertir timestamp a índice datetime
    df = df.set_index("timestamp")

    # 📌 Asegurar que `casa_id` es un string (por si hay valores enteros)
    df["casa_id"] = df["casa_id"].astype(str)

    # 📌 Rellenar valores nulos si existen
    df.fillna(method="ffill", inplace=True)

    # 📌 Guardar el dataset preprocesado en MinIO (como Parquet)
    output_path = "/mnt/data/consumo_preprocesado.parquet"
    df.to_parquet(output_path)

    # 📌 Registrar el dataset preprocesado en MLRun
    context.log_dataset("preprocessed_data", df=df, format="parquet", artifact_path=context.artifact_path)

    print("✅ Dataset preprocesado y guardado en MinIO correctamente.")

    return output_path

# 🔹 Ejecutar la función si se corre directamente
if __name__ == "__main__":
    ctx = mlrun.get_or_create_ctx("preprocess_consumo")
    preprocess(ctx)
