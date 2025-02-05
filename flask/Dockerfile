# Archivo: Dockerfile
FROM python:3.9-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo de dependencias e instala MLflow (y lo que necesites)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expone el puerto 5000 (que usará mlflow models serve)
EXPOSE 5000

# Al iniciar el contenedor, se ejecuta mlflow models serve usando la variable de entorno MLFLOW_MODEL_URI.
# Se utiliza --no-conda para evitar crear un entorno conda, lo que agiliza el arranque.
CMD ["sh", "-c", "mlflow models serve -m $MLFLOW_MODEL_URI -p 5000 --no-conda"]
