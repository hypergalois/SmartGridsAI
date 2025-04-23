import pandas as pd

def winsorize(s, lower_pct=0.01, upper_pct=0.99):
    low, high = s.quantile([lower_pct, upper_pct])
    return s.clip(low, high)

def load_and_clean(path='dataset2.csv'):
    df = pd.read_csv(path, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    # Quitar outliers extremos
    df['precio_electricidad_MW'] = winsorize(df['precio_electricidad_MW'])
    df['demanda_MW']            = winsorize(df['demanda_MW'])
    return df

if __name__ == "__main__":
    df = load_and_clean()
    df.to_parquet('data/cleaned.parquet')
