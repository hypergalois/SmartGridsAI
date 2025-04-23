import numpy as np
import pandas as pd

def make_features(df):
    # calendario cíclico
    df['hour']       = df.index.hour
    df['sin_h']      = np.sin(2*np.pi*df['hour']/24)
    df['cos_h']      = np.cos(2*np.pi*df['hour']/24)
    df['dow']        = df.index.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['month']      = df.index.month
    df['sin_m']      = np.sin(2*np.pi*(df['month']-1)/12)
    df['cos_m']      = np.cos(2*np.pi*(df['month']-1)/12)
    df['doy']        = df.index.dayofyear
    df['sin_doy']    = np.sin(2*np.pi*(df['doy']-1)/365)
    df['cos_doy']    = np.cos(2*np.pi*(df['doy']-1)/365)

    # rolling proxies
    df['price_roll3h']  = df['precio_electricidad_MW'].rolling(3).mean()
    df['price_roll24h'] = df['precio_electricidad_MW'].rolling(24).mean()

    # lags
    for lag in [1,24,168]:
        df[f'price_lag{lag}'] = df['precio_electricidad_MW'].shift(lag)
    df['price_lag8760'] = df['precio_electricidad_MW'].shift(24*365)

    df['demand_lag168']     = df['demanda_MW'].shift(168)
    df['gen_ren_lag168']    = df['generacion_renovable_MW'].shift(168)
    df['gen_nonren_lag168'] = df['generacion_no_renovable_MW'].shift(168)
    df['temp_lag168']       = df['temperatura_media'].shift(168)

    return df.dropna()

if __name__ == "__main__":
    df = pd.read_parquet('data/cleaned.parquet')
    df_feat = make_features(df)
    df_feat.to_parquet('data/features.parquet')
import numpy as np
import pandas as pd

def make_features(df):
    # calendario cíclico
    df['hour']       = df.index.hour
    df['sin_h']      = np.sin(2*np.pi*df['hour']/24)
    df['cos_h']      = np.cos(2*np.pi*df['hour']/24)
    df['dow']        = df.index.dayofweek
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    df['month']      = df.index.month
    df['sin_m']      = np.sin(2*np.pi*(df['month']-1)/12)
    df['cos_m']      = np.cos(2*np.pi*(df['month']-1)/12)
    df['doy']        = df.index.dayofyear
    df['sin_doy']    = np.sin(2*np.pi*(df['doy']-1)/365)
    df['cos_doy']    = np.cos(2*np.pi*(df['doy']-1)/365)

    # rolling proxies
    df['price_roll3h']  = df['precio_electricidad_MW'].rolling(3).mean()
    df['price_roll24h'] = df['precio_electricidad_MW'].rolling(24).mean()

    # lags
    for lag in [1,24,168]:
        df[f'price_lag{lag}'] = df['precio_electricidad_MW'].shift(lag)
    df['price_lag8760'] = df['precio_electricidad_MW'].shift(24*365)

    df['demand_lag168']     = df['demanda_MW'].shift(168)
    df['gen_ren_lag168']    = df['generacion_renovable_MW'].shift(168)
    df['gen_nonren_lag168'] = df['generacion_no_renovable_MW'].shift(168)
    df['temp_lag168']       = df['temperatura_media'].shift(168)

    return df.dropna()

if __name__ == "__main__":
    df = pd.read_parquet('data/cleaned.parquet')
    df_feat = make_features(df)
    df_feat.to_parquet('data/features.parquet')
