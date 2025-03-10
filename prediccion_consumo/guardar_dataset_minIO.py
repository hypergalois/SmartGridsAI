import mlrun
import pandas as pd

def upload_dataset(context):
    """
    Carga un dataset local y lo guarda en MinIO dentro del bucket 'smartgrids_bucket' en MLRun.
    """

    # 📌 Ruta del archivo local a subir
    dataset_path = "consumo_ts_parte_0.csv"

    # 📌 Cargar el dataset
    df = pd.read_csv(dataset_path)
    
     # 📌 Forzar el tipo de `casa_id` a string para evitar errores en el Feature Store
    df["casa_id"] = df["casa_id"].astype(str)

    # 📌 Obtener el proyecto (si no existe, lo crea)
    project = mlrun.get_or_create_project("smartgrids", "./smartgrids")

    # 📌 Definir la ruta de almacenamiento en el nuevo bucket de MinIO
    artifact_path = f"s3://smartgrids-bucket/datasets/"

    # 📌 Guardar el dataset en MinIO dentro del bucket específico
    context.log_dataset(
        "consumo_electrico", 
        df=df, 
        format="csv", 
        artifact_path=artifact_path  # 📌 Ruta personalizada en el nuevo bucket
    )

    print(f"✅ Dataset guardado en MinIO dentro del bucket 'smartgrids_bucket' y proyecto '{project.name}'.")

# 🔹 Ejecutar la función asegurando que se usa el contexto correcto
if __name__ == "__main__":
    ctx = mlrun.get_or_create_ctx("upload_dataset_consumo")
    upload_dataset(ctx)
