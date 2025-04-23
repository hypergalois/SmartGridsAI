import pandas as pd
import numpy as np
import joblib

def rolling_mean(series, ts, window):
    vals = [series.get(ts - pd.Timedelta(hours=i), np.nan) for i in range(1, window+1)]
    return np.nanmean(vals)

if __name__=="__main__":
    df = pd.read_parquet('data/features.parquet')  # contiene histórico completo
    model = joblib.load('models/model_precio_lgbm.pkl')
    features = model.booster_.feature_name()

    # índice de forecast
    h0 = df.index.max() + pd.Timedelta(hours=1)
    h1 = h0 + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    future_idx = pd.date_range(h0,h1,freq='H')

    Xf = pd.DataFrame(index=future_idx)
    # repetir calendario y festivos...
    Xf['hour']  = Xf.index.hour
    Xf['sin_h'] = np.sin(2*np.pi*Xf['hour']/24)
    Xf['cos_h'] = np.cos(2*np.pi*Xf['hour']/24)
    Xf['dow']   = Xf.index.dayofweek
    Xf['is_weekend'] = Xf['dow'].isin([5,6]).astype(int)
    Xf['month'] = Xf.index.month
    Xf['sin_m'] = np.sin(2*np.pi*(Xf['month']-1)/12)
    Xf['cos_m'] = np.cos(2*np.pi*(Xf['month']-1)/12)
    Xf['doy']   = Xf.index.dayofyear
    Xf['sin_doy'] = np.sin(2*np.pi*(Xf['doy']-1)/365)
    Xf['cos_doy'] = np.cos(2*np.pi*(Xf['doy']-1)/365)
    Xf['festivo'] = Xf.index.strftime('%d-%m').isin(df.index.strftime('%d-%m')).astype(int)

    # lags & rollings
    src = df['precio_electricidad_MW']
    for lag in [1,24,168]:
        Xf[f'price_lag{lag}'] = [src.get(ts - pd.Timedelta(hours=lag),np.nan) for ts in Xf.index]
    Xf['price_lag8760'] = [src.get(ts - pd.Timedelta(hours=8760),np.nan) for ts in Xf.index]

    Xf['price_roll3h']  = [ rolling_mean(src,ts,3)  for ts in Xf.index ]
    Xf['price_roll24h'] = [ rolling_mean(src,ts,24) for ts in Xf.index ]

    # proxies de demanda/generation/temp
    for lag, col in [(168,'demanda_MW'),(168,'generacion_renovable_MW'),
                     (168,'generacion_no_renovable_MW'),(168,'temperatura_media')]:
        Xf[f'{col}_lag168'] = [df[col].get(ts-pd.Timedelta(hours=168),np.nan) for ts in Xf.index]
        Xf[col] = Xf[f'{col}_lag168']

    # rellenar y predecir
    Xf = Xf[features].ffill().bfill()
    y_fut = model.predict(Xf)

    pd.Series(y_fut, index=future_idx).to_csv('data/forecast_7d.csv')
