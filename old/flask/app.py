#!/usr/bin/env python3
# Archivo: app.py
# Líneas importantes:
#   Línea 8: Se obtiene la variable de entorno MLFLOW_MODEL_URI. Si no está definida, se usa un valor por defecto.
#   Línea 9: Se imprime la URI para depuración.
#   Línea 10: Se carga el modelo de MLflow.
import os
import mlflow.pyfunc
from flask import Flask, request, jsonify

app = Flask(__name__)

# Obtener la URI del modelo desde la variable de entorno
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/default_model/Production")
print(f"[INFO] Cargando modelo desde: {MODEL_URI}")

# Cargar el modelo con mlflow
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado correctamente"}), 500

    try:
        # Se espera que la petición tenga un JSON con los datos de entrada
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Error al leer el JSON de entrada", "detalles": str(e)}), 400

    try:
        # Realizamos la predicción. Se asume que el modelo espera un DataFrame o un array.
        predictions = model.predict(data)
        # Se convierte a lista para asegurarse de que sea serializable a JSON.
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": "Error durante la predicción", "detalles": str(e)}), 500

if __name__ == '__main__':
    # La app se ejecuta en todas las interfaces en el puerto 5000
    app.run(host='0.0.0.0', port=5000)
