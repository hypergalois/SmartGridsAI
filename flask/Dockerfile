# Archivo: Dockerfile
FROM python:3.9-slim

# Definir el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias y luego instalarlo
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código de la aplicación
COPY app.py /app/app.py

# Exponer el puerto 5000 (el que usa la app)
EXPOSE 5000

# Ejecutar la aplicación con gunicorn para producción
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
