kubectl version --client

helm version

kubectl create namespace mlrun

helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo update

kubectl --namespace mlrun create secret docker-registry registry-credentials \
  --docker-server=https://registry.hub.docker.com/ \
  --docker-username=<tu-usuario-dockerhub> \
  --docker-password=<tu-password> \
  --docker-email=<tu-email>

helm --namespace mlrun install mlrun-ce \
  --wait \
  --timeout 960s \
  --set global.registry.url=index.docker.io/<tu-usuario-dockerhub> \
  --set global.registry.secretName=registry-credentials \
  --set global.externalHostAddress=localhost \
  --set nuclio.dashboard.externalIPAddresses="{127.0.0.1}" \
  mlrun-ce/mlrun-ce

kubectl -n mlrun get pods
