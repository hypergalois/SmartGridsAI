import mlrun
from mlrun.platforms import auto_mount
import os

# Crear o cargar el proyecto MLRun en una carpeta específica
project = mlrun.get_or_create_project(
    name="smartgrids",
    context="./smartgrids"  # 📌 Especifica la carpeta raíz del proyecto
)

# Definir la función de preprocesamiento
preprocess = project.set_function(
    name="preprocess",
    kind="job",
    image="mlrun/mlrun",
    handler="SmartGridsAI.prediccion_consumo.preprocesamiento.preprocess:preprocess",
).apply(auto_mount)

# Definir la función de entrenamiento del modelo Prophet
train_model_prophet = project.set_function(
    name="train_model_prophet",
    kind="job",
    image="mlrun/mlrun",
    handler="SmartGridsAI.prediccion_consumo.models.train_model_prophet:train_model",
).apply(auto_mount)

# 📌 Registrar el workflow en MLRun
project.set_workflow(
    name="smartgrids_pipeline_consumo",
    workflow_path=os.path.abspath("pipeline_workflow.py")
)

# Guardar el proyecto en MLRun
project.save()
print("✅ Pipeline corregido con preprocesamiento y modelo Prophet organizado en `smartgrids` creado y guardado en MLRun.")
