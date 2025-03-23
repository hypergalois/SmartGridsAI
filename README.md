# 🔌 SmartGridsAI: Predicción de Consumo Eléctrico con MLOps

Este proyecto predice el consumo eléctrico mediante modelos de machine learning, incluyendo Prophet, ARIMA y N-HiTS. Usa **MLRun** para MLOps y **Kubernetes** para el despliegue del modelo.

## 🚀 Características

- Predicción de consumo eléctrico con diferentes modelos.
- Pipelines orquestados con MLRun.
- Despliegue de modelos en Kubernetes.
- CI/CD automático con GitHub Actions.
- API REST para inferencia en tiempo real.

---

## 📁 Estructura de Carpetas

SmartGridsAI/
├── functions/ # Funciones para MLRun
│ ├── preprocess-data/
│ │ └── preprocess.py
│ ├── train-prophet/
│ │ └── train.py
│ ├── train-arima/
│ │ └── train.py
│ ├── train-nhits/
│ │ └── train.py
│ ├── evaluate-model/
│ │ └── evaluate.py
│ └── predict/
│ └── predict.py
│
├── notebooks/
│ ├── final/
│ │ └── prophet_avanzado.ipynb
│ ├── experiments/
│ │ ├── Arima.ipynb
│ │ ├── Nhits.ipynb
│ │ ├── prophet_simple.ipynb
│ │ └── pythorchPrueba.ipynb
│ └── utils/
│ └── creaciondatasetmock.ipynb
│
├── deploy/
│ ├── mlflow_deployment_template.yaml
│ ├── mlflow_deployment.yaml
│ ├── update_mlserver_deployment.py
│ └── generate_mock_dataset.py
│
├── docker/
│ ├── Dockerfile.mlflow-serve
│ └── Dockerfile.flask-api
│
├── api/
│ ├── app.py
│ └── requirements.txt
│
├── pipelines/
│ └── training_pipeline.py
│
├── .github/
│ └── workflows/
│ └── deploy.yml
│
├── data/
│ └── consumo_ts_parte_0.csv
│
├── artifacts/
├── project.yaml
├── README.md
├── .gitignore
└── docker-compose.yml

---

## ⚙️ Requisitos

- Python 3.9+
- MLRun (recomendado en contenedor)
- Docker
- Kubernetes (Minikube o clúster en la nube)
- MLflow (opcional si usas `mlflow models serve`)

---

## 🏗️ MLRun Pipelines

### Crear y registrar el proyecto MLRun

```python
mlrun project ./ -n smartgrids
```

### Ejecutar el pipeline de entrenamiento

```python
mlrun run pipeline -p file_path=data/consumo_ts_parte_0.csv
```

---

## 🐳 Docker y Despliegue manual en Kubernetes

### Construir y subir imagen Docker

```python
docker build -t <dockerhub_user>/mlflow-model-server:latest .
docker push <dockerhub_user>/mlflow-model-server:latest
```

### Actualizar YAML de despliegue con la URI del modelo

```python
python deploy/update_mlserver_deployment.py “models:/prophet_model/Production”
```

### Aplicar el deployment en Kubernetes

```python
kubectl apply -f deploy/mlflow_deployment.yaml
kubectl rollout status deployment/mlflow-model-server
```

---

## 🌐 Probar el endpoint de inferencia

```python
curl –request POST http://:/predict
–header ‘Content-Type: application/json’
–data ‘{“ds”: “2025-03-01”}’
```

---

## ⚡ CI/CD Automático con GitHub Actions

### Descripción del flujo

1. Push a la rama `main`.
2. GitHub Actions construye la imagen Docker.
3. Push de la imagen a DockerHub.
4. Actualiza el `mlflow_deployment.yaml`.
5. Aplica el deployment en Kubernetes.

### Desplegar manualmente desde GitHub Actions

- Ir a `Actions` → `Deploy MLflow Model Server` → `Run workflow`.

---

## 👨‍💻 Colaboradores

- etc...

---

## 📄 Licencia

MIT License
