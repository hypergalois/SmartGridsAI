import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from tqdm import trange

# Comprobacion gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# --- Nuevo: importar wandb ---
import wandb

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
state_columns = ['consumo_kWh', 'produccion_kWh', 'coste_euros']

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
            'value': 10000
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}


# ==============================================
# 5) RED NEURONAL (DQN)
# ==============================================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==============================================
# 6) AGENTE DQN
# ==============================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device('cpu')  # Forzamos CPU
        self.model.to(self.device)
    
    def act(self, state, epsilon=0.1):
        # Política epsilon-greedy
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_t)
        action = torch.argmax(q_values).item()
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Retorna el 'loss' promedio de este batch para poder loguearlo en wandb.
        """
        if len(self.memory) < BATCH_SIZE:
            return None
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        # Q(s,a) actual
        current_Q = self.model(states).gather(1, actions).squeeze()

        # Q(s', a') futuro
        next_Q = self.model(next_states).max(1)[0]
        target_Q = rewards + GAMMA * next_Q * (~dones)

        loss = self.loss_fn(current_Q, target_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# ==============================================
# 7) FUNCIÓN DE RECOMPENSA (EJEMPLO SIMPLE)
# ==============================================
def calcular_recompensa(row, action):
    """
    row: Fila del DataFrame con [consumo_kWh, produccion_kWh, precio_electricidad] o lo que uses.
    action: 0=Mantener, 1=Comprar, 2=Vender
    """
    precio = row['coste_euros']
    consumo = row['consumo_kWh']
    produccion = row['produccion_kWh']

    neto = produccion - consumo  # >0 excedente, <0 déficit

    if action == 0:  # Mantener
        reward = -abs(neto) * 0.1
    elif action == 1:  # Comprar
        if neto > 0:
            reward = -1.0
        else:
            reward = 50 / precio  # Cuanto más bajo el precio, mejor
    else:  # Vender
        if neto < 0:
            reward = -1.0
        else:
            reward = precio / 50.0  # Cuanto más alto el precio, mejor
    return reward

# ==============================================
# 8) INICIALIZAR W&B (PROJECT="SmartGrids")
# ==============================================
wandb.init(
    project="SmartGrids", 
    name="DQN-CompraVenta-CasaUnica",
    config={
        "episodes": NUM_EPISODES,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA
    }
)
config = wandb.config

# ==============================================
# 9) BUCLE DE ENTRENAMIENTO (SÓLO CASA 3234)
# ==============================================
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

epsilon = EPSILON_START

# Filtramos y usamos sólo df_casa (que ya tiene id=3234)
# Asegúrate de que df_casa tenga suficientes filas
print(f"Entrenando solo para la casa {casa_id_objetivo}, con {len(df_casa)} registros.")

for episode in trange(NUM_EPISODES, desc="Entrenando"):
    total_reward = 0.0
    # Recorremos cada fila de la casa como "steps"
    print(f"Entrenando episodio {episode+1}...")
    for i in range(len(df_casa) - 1):
        current_row = df_casa.iloc[i]
        state = current_row[state_columns].values.astype(np.float32)

        action = agent.act(state, epsilon)
        reward = calcular_recompensa(current_row, action)
        total_reward += reward

        next_row = df_casa.iloc[i+1]
        next_state = next_row[state_columns].values.astype(np.float32)

        done = (i == len(df_casa) - 2)

        agent.remember(state, action, reward, next_state, done)

        # (Opcional) Replay a cada step. 
        # Si quieres esperar al final de la "época" puedes moverlo fuera
        loss_val = agent.replay()

        wandb.log({
            "step_reward": reward,
            "epsilon": epsilon,
            "loss": loss_val if loss_val is not None else 0.0
        })

    # Reducimos epsilon para disminuir exploración
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Registramos métricas en Weights & Biases
    wandb.log({
        "episode": episode + 1,
        "total_reward": total_reward,
        "epsilon": epsilon,
        "loss": loss_val if loss_val is not None else 0.0
    })

    # Impresión periódica
    if (episode+1) % 5 == 0:
        print(f"[Episodio {episode+1}] Recompensa total: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Entrenamiento finalizado.")
wandb.finish()

# Guardar el modelo 
torch.save(agent.model.state_dict(), "dqn_model.pth")