import mlrun

# Definir nombre del proyecto
project_name = "smartgrids"

# Definir la ruta donde se almacenarán los artefactos
artifact_path = "s3://smartgrids-bucket"

# Definir el repositorio de código en GitHub
git_repo = "https://github.com/hypergalois/SmartGridsAI"

# Crear el proyecto en MLRun (o cargarlo si ya existe)
project = mlrun.get_or_create_project(
    name=project_name,
    context="./",  # Contexto local donde se clona el repo
    user_project=False,
    remote=git_repo
)

# Configurar MinIO como bucket para almacenar artefactos
project.set_artifact_path(artifact_path)

# Imprimir información del proyecto
print(f"Proyecto '{project_name}' configurado con éxito!")
print(f"Repositorio Git: {git_repo}")
print(f"Ruta de artefactos: {artifact_path}")
