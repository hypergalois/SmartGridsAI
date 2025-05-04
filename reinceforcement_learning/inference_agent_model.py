import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ======= CONFIGURACIÓN =======
MODEL_PATH = "dqn_model_DQN-Sweep.pth"  # Cambia si tienes un nombre específico
CASA_ID = 3234
BATERIA_CAPACIDAD = 13.5
BATERIA_MAX_POTENCIA = 5.0
BATERIA_EFICIENCIA = 0.95

# === CARGA DEL MODELO ===
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, num_layers=2):
        super().__init__()
        layers = [nn.Linear(state_size, hidden_size), nn.LeakyReLU(0.01)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

STATE_COLUMNS = ['consumo_kWh', 'produccion_kWh', 'coste_euros', 'irradiancia_W_m2', 'num_placas', 'humedad']
STATE_SIZE = len(STATE_COLUMNS) + 1  # + battery_soc
ACTION_SIZE = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DQN(STATE_SIZE, ACTION_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === CARGA DE DATOS ===
df_consumo = pd.read_csv("datasets/dataset_consumo.csv")
df_produccion = pd.read_csv("datasets/dataset_produccion.csv")
df = pd.merge(df_consumo, df_produccion, on=["datetime", "id_casa"], how="inner")
df.sort_values(['id_casa', 'datetime'], inplace=True)
df_casa = df[df['id_casa'] == CASA_ID].copy().reset_index(drop=True)

# Tomamos 1 día (24 horas)
df_dia = df_casa.head(24).copy()

# === INFERENCIA HORA A HORA ===
battery_soc = BATERIA_CAPACIDAD / 2
beneficio_total = 0

acciones = {0: "Mantener", 1: "Comprar", 2: "Vender"}

print(f"\nInferencia para la casa {CASA_ID}, batería inicial al 50% ({battery_soc:.2f} kWh)\n")
print("{:<8} {:<10} {:<10} {:<12} {:<15} {:<10}".format("Hora", "Acción", "Precio", "Batería(kWh)", "Beneficio (€)", "SOC %"))

for i in range(len(df_dia)):
    row = df_dia.iloc[i]
    state_vals = row[STATE_COLUMNS].values.astype(np.float32)
    state = np.append(state_vals, battery_soc / BATERIA_CAPACIDAD)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    precio = row["coste_euros"]
    beneficio = 0

    if action == 1:  # Comprar
        energia = min(BATERIA_MAX_POTENCIA, BATERIA_CAPACIDAD - battery_soc)
        if energia > 0:
            battery_soc += energia * BATERIA_EFICIENCIA
            beneficio = -energia * precio
    elif action == 2:  # Vender
        energia = min(BATERIA_MAX_POTENCIA, battery_soc)
        if energia > 0:
            battery_soc -= energia / BATERIA_EFICIENCIA
            beneficio = energia * precio

    beneficio_total += beneficio

    hora = pd.to_datetime(row['datetime']).strftime('%H:%M')
    print("{:<8} {:<10} {:<10.2f} {:<12.2f} {:<15.2f} {:<10.1f}".format(
        hora, acciones[action], precio, battery_soc, beneficio, battery_soc / BATERIA_CAPACIDAD * 100))

print(f"\nBeneficio total del día: {beneficio_total:.2f} €")
