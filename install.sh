#!/bin/bash

# VARIABLES - Cambia el valor de DOCKERHUB_USER si es necesario
DOCKERHUB_USER="user"

# Crear namespace
kubectl create namespace mlrun

# Añadir repositorio de Helm de MLRun CE
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update

# Crear el secret del docker registry (Docker Hub)
kubectl --namespace mlrun create secret docker-registry registry-credentials \
  --docker-server=https://registry.hub.docker.com/ \
  --docker-username=$DOCKERHUB_USER \
  --docker-password=<TU_ACCESS_TOKEN_O_PASSWORD> \
  --docker-email=<TU_EMAIL>

# Instalar MLRun CE en versión ligera
helm --namespace mlrun install mlrun-ce \
  --wait \
  --timeout 960s \
  --set global.registry.url=index.docker.io/$DOCKERHUB_USER \
  --set global.registry.secretName=registry-credentials \
  --set global.externalHostAddress=localhost \
  --set nuclio.dashboard.externalIPAddresses="{127.0.0.1}" \
  --set pipelines.enabled=false \
  --set sparkOperator.enabled=false \
  --set kube-prometheus-stack.enabled=false \
  mlrun-ce/mlrun-ce

# Mostrar estado de los pods después de la instalación
kubectl -n mlrun get pods