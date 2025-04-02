from kfp import dsl
import mlrun

@dsl.pipeline(name="smartgrids_pipeline_consumo", description="Pipeline con preprocesado y modelo Prophet")
def pipeline():
    # 📌 URI del dataset en MinIO
    dataset_uri = "store://datasets/smartgrids/guardar-dataset-minio_consumo_electrico#0:latest"

    # 📌 Paso 1: Preprocesamiento
    preprocess_task = mlrun.run_function(
        "preprocess",
        inputs={"dataset": dataset_uri},
        outputs=["preprocessed_data"]
    )

    # 📌 Paso 2: Entrenamiento del modelo Prophet
    train_task = mlrun.run_function(
        "train_model_prophet",
        inputs={"dataset": preprocess_task.outputs["preprocessed_data"]}
    )

# Guardar el pipeline
pipeline()
