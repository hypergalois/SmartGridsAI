#!/usr/bin/env python3
# Archivo: update_mlserver_deployment.py

import os
import sys
import yaml


def update_yaml(template_file, output_file, model_uri):
    try:
        with open(template_file, "r") as f:
            # Cargamos todos los documentos YAML (Deployment y Service)
            docs = list(yaml.safe_load_all(f))
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo {template_file}: {e}")
        sys.exit(1)

    # Buscamos en cada documento el Deployment para actualizar la variable de entorno
    for doc in docs:
        if doc.get("kind") == "Deployment":
            try:
                containers = doc["spec"]["template"]["spec"]["containers"]
                for container in containers:
                    if container.get("name") == "mlflow-model-server":
                        env_vars = container.get("env", [])
                        updated = False
                        for env_var in env_vars:
                            if env_var.get("name") == "MLFLOW_MODEL_URI":
                                env_var["value"] = model_uri
                                updated = True
                        if not updated:
                            env_vars.append({"name": "MLFLOW_MODEL_URI", "value": model_uri})
                        container["env"] = env_vars
            except Exception as e:
                print(f"[ERROR] Error al actualizar el YAML: {e}")
                sys.exit(1)

    try:
        with open(output_file, "w") as f:
            yaml.safe_dump_all(docs, f)
        print(f"[INFO] Archivo YAML actualizado guardado en {output_file} con MLFLOW_MODEL_URI = {model_uri}")
    except Exception as e:
        print(f"[ERROR] No se pudo escribir el archivo {output_file}: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Uso: python update_mlserver_deployment.py <MLFLOW_MODEL_URI>")
        sys.exit(1)

    model_uri = sys.argv[1]
    template_file = "mlflow_deployment_template.yaml"  # Asegúrate de que este archivo exista.
    output_file = "mlflow_deployment.yaml"  # Este es el archivo que se aplicará en Kubernetes.
    update_yaml(template_file, output_file, model_uri)


if __name__ == "__main__":
    main()
