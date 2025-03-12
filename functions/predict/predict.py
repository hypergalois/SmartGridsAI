import mlrun
import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, body: dict):
        """
        Realiza una predicción a partir de una fecha.
        Ejemplo entrada: { "ds": "2025-03-01" }
        """
        df_input = pd.DataFrame([body])
        forecast = self.model.predict(df_input)

        return {"yhat": forecast['yhat'].iloc[0]}