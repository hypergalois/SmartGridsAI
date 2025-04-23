import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def smape(a, f):
    return 100/len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))

if __name__ == "__main__":
    df = pd.read_parquet('data/features.parquet')
    features = [  # idéntico a tu lista
        'demanda_MW','generacion_renovable_MW','generacion_no_renovable_MW',
        'temperatura_media','festivo',
        'sin_h','cos_h','dow','is_weekend',
        'sin_m','cos_m','sin_doy','cos_doy',
        'price_roll3h','price_roll24h',
        'price_lag1','price_lag24','price_lag168','price_lag8760',
        'demand_lag168','gen_ren_lag168','gen_nonren_lag168',
        'temp_lag168'
    ]
    X = df[features]
    y = df['precio_electricidad_MW']

    split_date = df.index.max() - pd.Timedelta(days=7)
    X_train = X.loc[:split_date];   y_train = y.loc[:split_date]
    X_test  = X.loc[split_date+pd.Timedelta(hours=1):]
    y_test  = y.loc[split_date+pd.Timedelta(hours=1):]

    param_grid = {
      'n_estimators': [200,500],
      'max_depth':    [6,10],
      'learning_rate':[0.05,0.1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    gscv = GridSearchCV(
      LGBMRegressor(random_state=42),
      param_grid, cv=tscv,
      scoring='neg_mean_absolute_error',
      n_jobs=-1, verbose=1
    )
    gscv.fit(X_train,y_train)
    model = gscv.best_estimator_
    joblib.dump(model,'models/model_precio_lgbm.pkl')

    y_pred = model.predict(X_test)
    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test,y_pred))
    mape  = mean_absolute_percentage_error(y_test,y_pred)
    smape_val = smape(y_test.values,y_pred)

    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3%}, SMAPE: {smape_val:.3f}%")
