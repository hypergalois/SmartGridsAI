# 🚀 Instalación de MLRun Community Edition en Local (Docker Desktop + Kubernetes)

## ✅ Requisitos previos

Asegúrate de tener lo siguiente instalado y configurado antes de comenzar:

- **Docker Desktop** con **Kubernetes** activado.

  - Activar en: `Preferences ➜ Kubernetes ➜ Enable Kubernetes`.
  - Aplicar cambios con `Apply & Restart`.

- **kubectl** (suele venir con Docker Desktop).

  ```bash
  kubectl version --client
  ```

- **Helm** (versión 3.6 o superior).
  ```bash
  helm version
  ```

## ✅ Configurar recursos en Docker Desktop

- **CPU**: mínimo 4 (ideal 6-8).
- **Memoria RAM**: mínimo 12 GB (ideal 16 GB).

Configura en `Docker Desktop ➜ Settings ➜ Resources ➜ Advanced`.

---

## ✅ Paso 1. Crear el namespace en Kubernetes

```bash
kubectl create namespace mlrun
```

---

## ✅ Paso 2. Añadir el repositorio de Helm de MLRun CE

```bash
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update
```

---

## ✅ Paso 3. Crear el secret de Docker Registry

Esto es necesario para que MLRun construya imágenes.

```bash
kubectl --namespace mlrun create secret docker-registry registry-credentials \
  --docker-server=https://registry.hub.docker.com/ \
  --docker-username=<tu-usuario-dockerhub> \
  --docker-password=<tu-password-o-access-token> \
  --docker-email=<tu-email>
```

> ⚠️ Si tienes 2FA en Docker Hub, usa un **Access Token** en lugar de la contraseña normal.  
> [Cómo generar un Access Token en Docker Hub](https://hub.docker.com/settings/security).

---

## ✅ Paso 4. Instalar MLRun CE en Kubernetes (versión ligera y estable)

```bash
helm --namespace mlrun install mlrun-ce \
  --wait \
  --timeout 960s \
  --set global.registry.url=index.docker.io/<tu-usuario-dockerhub> \
  --set global.registry.secretName=registry-credentials \
  --set global.externalHostAddress=localhost \
  --set nuclio.dashboard.externalIPAddresses="{127.0.0.1}" \
  --set pipelines.enabled=false \
  --set sparkOperator.enabled=false \
  --set kube-prometheus-stack.enabled=false \
  mlrun-ce/mlrun-ce
```

> Reemplaza `<tu-usuario-dockerhub>` con tu nombre de usuario de Docker Hub.

---

## ✅ Servicios levantados (por defecto)

| Servicio         | URL                    | Usuario/Pass (si aplica)           |
| ---------------- | ---------------------- | ---------------------------------- |
| Jupyter Notebook | http://localhost:30040 |                                    |
| Nuclio Dashboard | http://localhost:30050 |                                    |
| MLRun UI         | http://localhost:30060 |                                    |
| MLRun API        | http://localhost:30070 |                                    |
| MinIO UI         | http://localhost:30090 | usuario: `minio`, pass: `minio123` |

---

## ✅ Verificar estado de los pods

```bash
kubectl -n mlrun get pods
```

Todos deberían estar en estado `Running`.

---

## ✅ Primeros pasos en MLRun

1. Accede a **Jupyter Notebook** ➜ [http://localhost:30040](http://localhost:30040).
2. Abre y ejecuta el notebook de ejemplo:
   ```
   /examples/mlrun_basics.ipynb
   ```
3. Guarda tus datos en la carpeta `/data` para evitar pérdidas al reiniciar.

---

## ✅ Cómo **apagar** y **levantar** MLRun

### OPCIÓN 1: Usar Docker Desktop

- **Apagar MLRun**: Cierra Docker Desktop (apaga Kubernetes).
- **Levantar MLRun**: Abre Docker Desktop ➜ Kubernetes se inicia ➜ MLRun arranca solo.
- Comprobar estado:
  ```bash
  kubectl -n mlrun get pods
  ```

---

### OPCIÓN 2: Control fino con `kubectl`

#### Apagar (detener todos los pods sin desinstalar):

```bash
kubectl -n mlrun scale deployment --all --replicas=0
```

#### Encender (volver a levantar los pods):

```bash
kubectl -n mlrun scale deployment --all --replicas=1
```

---

## ✅ Cómo **desinstalar** completamente MLRun CE

1. Desinstalar el Helm release:

   ```bash
   helm --namespace mlrun uninstall mlrun-ce
   ```

2. Borrar el namespace (opcional si quieres limpiar del todo):

   ```bash
   kubectl delete namespace mlrun
   ```

3. Si el namespace se queda colgado:

   ```bash
   kubectl delete namespace mlrun --grace-period=0 --force
   ```

4. Limpiar PVCs (opcional, destruye datos persistentes):
   ```bash
   kubectl get pvc -n mlrun
   kubectl delete pvc <nombre-pvc>
   ```

---

## ✅ Reinstalar MLRun CE tras desinstalación

```bash
kubectl create namespace mlrun

helm --namespace mlrun install mlrun-ce \
  --wait \
  --timeout 960s \
  --set global.registry.url=index.docker.io/<tu-usuario-dockerhub> \
  --set global.registry.secretName=registry-credentials \
  --set global.externalHostAddress=localhost \
  --set nuclio.dashboard.externalIPAddresses="{127.0.0.1}" \
  --set pipelines.enabled=false \
  --set sparkOperator.enabled=false \
  --set kube-prometheus-stack.enabled=false \
  mlrun-ce/mlrun-ce
```

---

## ✅ Recomendaciones

- **Subir recursos** en Docker Desktop ➜ CPUs: 6+, RAM: 12-16 GB.
- **Guardar los notebooks** en la carpeta `/data` dentro de Jupyter.
- **Control de acceso** ➜ Cambia las credenciales de MinIO si es necesario.

---

## ✅ Recursos de MLRun útiles

- [Documentación oficial de MLRun CE](https://docs.mlrun.org/en/latest/)
- [Ejemplos de notebooks MLRun](https://github.com/mlrun/mlrun/tree/development/examples)
- [Nuclio Serverless Functions](https://nuclio.io/)

---

## ✅ Happy MLOpsing! 🚀
