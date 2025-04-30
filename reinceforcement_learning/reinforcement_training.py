import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from tqdm import trange

# --- Nuevo: importar wandb ---
import wandb

print("Nombre de la GPU:", torch.cuda.get_device_name(0))
print("Número de GPUs disponibles:", torch.cuda.device_count())
print("Dispositivo actual:", torch.cuda.current_device())

# ==============================================
# 1) CARGAR TUS DATASETS REALES
# ==============================================
dataset_consumo = pd.read_csv('datasets/dataset_consumo.csv')       # Ajusta la ruta
dataset_produccion = pd.read_csv('datasets/dataset_produccion.csv') # Ajusta la ruta

# Suponemos que cada uno tiene columnas 'datetime', 'id_casa', y:
# - dataset_consumo => 'consumo_kWh'
# - dataset_produccion => 'produccion_kWh'
# Otras columnas también son bienvenidas, por ejemplo 'temperatura', etc.

# 1.1) Merge de ambos en un solo DataFrame (clave: 'datetime' + 'id_casa')
df = pd.merge(dataset_consumo, dataset_produccion, on=['datetime','id_casa'], how='inner')

# Ordenamos para evitar desorden temporal
df.sort_values(['id_casa', 'datetime'], inplace=True, ignore_index=True)

# ==============================================
# 2) FILTRAR A UNA SOLA CASA (id_casa = 3234)
# ==============================================
casa_id_objetivo = 3234
df_casa = df[df['id_casa'] == casa_id_objetivo].copy()
df_casa.sort_values('datetime', inplace=True, ignore_index=True)

# ==============================================
# 3) DEFINIR COLUMNAS DEL ESTADO
# ==============================================
# Suponiendo que tu DataFrame final tiene estas columnas
# (cambia 'coste_euros' por 'precio_electricidad' si corresponde)
state_columns = ['consumo_kWh', 'produccion_kWh', 'coste_euros', 'irradiancia_W_m2', 'num_placas', 'humedad']

STATE_SIZE = len(state_columns)

# ==============================================
# 4) HIPERPARÁMETROS RL
# ==============================================
ACTION_SIZE = 3   # 0: Mantener, 1: Comprar, 2: Vender
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
NUM_EPISODES = 50
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99


# ==============================================
# SWEEP CONFIG INTEGRADO EN EL SCRIPT
# ==============================================
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'total_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'gamma': {
            'min': 0.85,
            'max': 0.999
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'epsilon_decay': {
            'min': 0.90,
            'max': 0.99
        },
        'memory_size': {
            'value': 1000
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}

wandb.init(
    project="SmartGrids", 
    name="DQN-CompraVenta-CasaUnica",
    config={
        "gamma": GAMMA,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "hidden_size": 128,
        "num_layers": 2,
        "memory_size": MEMORY_SIZE,
        "epsilon_decay": EPSILON_DECAY
    }
)
config = wandb.config


# ==============================================
# 5) RED NEURONAL (DQN)
# ==============================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers):
        super().__init__()
        layers = [nn.Linear(state_size, hidden_size), nn.LeakyReLU(0.01)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(hidden_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==============================================
# 6) AGENTE DQN
# ==============================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=config.memory_size)
        self.model = DQN(state_size, action_size, config.hidden_size, config.num_layers)
        self.target_model = DQN(state_size, action_size, config.hidden_size, config.num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.update_target_network()
        self.update_counter = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(ACTION_SIZE)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < config.batch_size:
            return None
        batch = random.sample(self.memory, config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        current_Q = self.model(states).gather(1, actions).squeeze()
        next_Q = self.target_model(next_states).max(1)[0]
        target_Q = rewards + config.gamma * next_Q * (~dones)
        loss = self.loss_fn(current_Q, target_Q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.update_target_network()
        return loss.item()
# ==============================================
# 7) FUNCIÓN DE RECOMPENSA (EJEMPLO SIMPLE)
# ==============================================
def calcular_recompensa(row, action, battery_soc):
    precio = row['coste_euros']
    consumo = row['consumo_kWh']
    produccion = row['produccion_kWh']
    neto = produccion - consumo
    capacidad = 13.5
    max_potencia = 5.0
    eficiencia = 0.95
    reward = 0
    beneficio = 0

    if action == 0:  # mantener
        reward = -abs(neto) * 0.05

    elif action == 1:  # comprar
        energia_a_comprar = min(max_potencia, capacidad - battery_soc)
        if energia_a_comprar > 0:
            battery_soc += energia_a_comprar * eficiencia
            costo = energia_a_comprar * precio
            reward = -costo
            beneficio = -costo
        else:
            reward = -5

    elif action == 2:  # vender
        energia_a_vender = min(max_potencia, battery_soc)
        if energia_a_vender > 0:
            battery_soc -= energia_a_vender / eficiencia
            ingreso = energia_a_vender * precio
            reward = ingreso
            beneficio = ingreso
        else:
            reward = -5

    return reward, battery_soc, beneficio

# ==============================================
# 9) BUCLE DE ENTRENAMIENTO (SÓLO CASA 3234)
# ==============================================
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

epsilon = EPSILON_START

# Filtramos y usamos sólo df_casa (que ya tiene id=3234)
# Asegúrate de que df_casa tenga suficientes filas
print(f"Entrenando solo para la casa {casa_id_objetivo}, con {len(df_casa)} registros.")
print("Nombre de la GPU:", torch.cuda.get_device_name(0))
print("Número de GPUs disponibles:", torch.cuda.device_count())
print("Dispositivo actual:", torch.cuda.current_device())

for episode in trange(NUM_EPISODES, desc="Entrenando"):
    total_reward = 0.0
    beneficio_total = 0.0
    battery_soc = 6.75  # 50% cargada

    for i in range(len(df_casa) - 1):
        current_row = df_casa.iloc[i]
        next_row = df_casa.iloc[i + 1]

        state_base = current_row[state_columns].values.astype(np.float32)
        next_state_base = next_row[state_columns].values.astype(np.float32)

        # Normalizar battery_soc (0–1)
        state = np.append(state_base, battery_soc / 13.5)
        next_state = np.append(next_state_base, battery_soc / 13.5)

        action = agent.act(state, epsilon)
        reward, battery_soc, beneficio = calcular_recompensa(current_row, action, battery_soc)
        done = (i == len(df_casa) - 2)

        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        beneficio_total += beneficio

        loss = agent.replay()
        if loss:
            wandb.log({"loss": loss})

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    wandb.log({
        "episode": episode + 1,
        "total_reward": total_reward,
        "beneficio_euros": beneficio_total,
        "epsilon": epsilon
    })

    # Impresión periódica
    if (episode+1) % 5 == 0:
        print(f"[Episodio {episode+1}] Recompensa total: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Entrenamiento finalizado.")
wandb.finish()

# Guardar el modelo 
torch.save(agent.model.state_dict(), "dqn_model.pth")