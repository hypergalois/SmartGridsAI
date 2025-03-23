import mlrun
import pandas as pd
import joblib
from datetime import timedelta

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, body: dict):
        """
        Predice el valor futuro.
        Ejemplo entrada: { "ds": "2025-03-01" }
        """

        date = pd.to_datetime(body["ds"])

        # Prophet usa DataFrame, ARIMA solo n_periods
        if hasattr(self.model, 'predict'):
            df_input = pd.DataFrame({"ds": [date]})
            forecast = self.model.predict(df_input)
            yhat = forecast['yhat'].iloc[0]
        else:
            forecast = self.model.predict(n_periods=1)
            yhat = forecast[0]

        return {"ds": str(date), "yhat": yhat}