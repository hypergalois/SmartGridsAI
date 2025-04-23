import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_last7_and_forecast(df_hist, y_pred, future_idx):
    hist7 = df_hist.iloc[-7*24:]
    plt.figure(figsize=(12,4))
    plt.plot(hist7.index, hist7,      label='Real hist.',  color='tab:orange')
    plt.plot(hist7.index, y_pred,      label='Predicción hist.', color='tab:blue')
    plt.plot(future_idx, y_pred[-len(future_idx):], label='Forecast 7d', color='tab:green')
    plt.title('Últimos 7 días y siguiente forecast 7 días')
    plt.ylabel('€/MWh'); plt.xlabel('Fecha')
    plt.legend(); plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %Hh'))
    plt.xticks(rotation=45); plt.tight_layout()
    plt.show()

def plot_demand_price_temp(df):
    fig, ax1 = plt.subplots(figsize=(14,4))
    ax1.fill_between(df.index, df['demanda_MW'],
                     color='tab:orange', alpha=0.3, label='Demanda (MW)')
    ax1.set_ylabel('Demanda (MW)', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')

    ax2 = ax1.twinx()
    precio_smooth = df['precio_electricidad_MW'].rolling(24).mean()
    ax2.plot(df.index, precio_smooth,
             color='tab:blue', label='Precio €/MWh (24h rolling)')
    ax2.set_ylabel('Precio €/MWh', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.12))
    ax3.plot(df.index, df['temperatura_media'],
             color='tab:green', alpha=0.6, label='Temperatura (°C)')
    ax3.set_ylabel('Temp. (°C)', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # leyenda
    lines, labels = [], []
    for ax in (ax1,ax2,ax3):
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    ax1.legend(lines, labels, loc='upper left', ncol=3)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title("Relación Demanda, Precio y Temperatura (serie completa)")
    plt.tight_layout()
    plt.show()
