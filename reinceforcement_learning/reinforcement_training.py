import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from tqdm import trange

# ==============================================
# 1) GENERAR DATOS SINTÉTICOS PARA 100 CASAS
# ==============================================
num_casas = 100
horas_por_casa = 24  # Un día de ejemplo
n_rows = num_casas * horas_por_casa

df = pd.read_csv('dataset_d3_filtrado.csv')  # Cargar datos reales

# Ordenamos por id_casa y datetime para no mezclar
df.sort_values(['id_casa', 'datetime'], inplace=True, ignore_index=True)

# Podemos normalizar o escalar estas variables si hace falta
# (Aquí simplemente las dejamos así para el ejemplo)

# Definimos qué columnas componen nuestro "estado"
state_columns = ['consumo_kWh', 'produccion_kWh', 'precio_electricidad']

# El tamaño de estado es el número de columnas elegidas
STATE_SIZE = len(state_columns)

# ==============================================
# 2) DEFINIR PARÁMETROS DE RL
# ==============================================
ACTION_SIZE = 3  # 0: Mantener, 1: Comprar, 2: Vender
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
NUM_EPISODES = 50  # Ejemplo de episodios
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.95

# ==============================================
# 3) DEFINIR DQN (RED NEURONAL)
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
# 4) DEFINIR AGENTE DQN
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
        # Si no hay batch suficiente, no entrenamos
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Q(s,a)
        current_Q = self.model(states).gather(1, actions).squeeze()

        # max_a' Q(s',a')
        next_Q = self.model(next_states).max(1)[0]
        target_Q = rewards + GAMMA * next_Q * (~dones)

        loss = self.loss_fn(current_Q, target_Q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==============================================
# 5) FUNCIÓN DE RECOMPENSA (EJEMPLO SIMPLIFICADO)
# ==============================================
def calcular_recompensa(row, action):
    """
    row: Fila actual con consumo, produccion, precio, etc.
    action: 0=Mantener, 1=Comprar, 2=Vender

    - Compramos cuando el precio es bajo => reward + (inversamente al precio)
    - Vendemos cuando el precio es alto => reward + (precio)
    - Mantener => reward pequeño basado en equilibrio
    """
    precio = row['precio_electricidad']
    consumo = row['consumo_kWh']
    produccion = row['produccion_kWh']

    # EJEMPLO: la cantidad neta es produccion - consumo
    #  si >0 hay excedente, si <0 hay déficit
    neto = produccion - consumo

    # Caso 0: Mantener
    if action == 0:
        # premio por estar equilibrado
        reward = -abs(neto) * 0.1
    # Caso 1: Comprar
    elif action == 1:
        # penalización si neto ya era positivo
        if neto > 0:
            reward = -1.0
        else:
            # hipotético: si precio es bajo, sumamos inverso
            reward = 50 / precio  # más alto si precio es bajo
    # Caso 2: Vender
    else:
        # penalización si neto era negativo
        if neto < 0:
            reward = -1.0
        else:
            # beneficio si precio es alto
            reward = precio / 50.0

    return reward

# ==============================================
# 6) BUCLE PRINCIPAL DE ENTRENAMIENTO
# ==============================================
agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

epsilon = EPSILON_START

for episode in trange(NUM_EPISODES, desc="Entrenando"):
    total_reward = 0.0

    # Mezclamos el orden de las casas en cada episodio
    casas = df['id_casa'].unique()
    np.random.shuffle(casas)

    for casa_id in casas:
        # Filtramos datos de la casa y ordenamos por tiempo
        df_casa = df[df['id_casa'] == casa_id].sort_values('datetime')
        df_casa = df_casa.reset_index(drop=True)

        # Recorremos las filas de la casa
        for i in range(len(df_casa) - 1):
            # Estado actual
            current_row = df_casa.iloc[i]
            state = current_row[state_columns].values.astype(np.float32)

            # Acción del agente (política epsilon-greedy)
            action = agent.act(state, epsilon)

            # Calculamos la recompensa en base a la acción
            reward = calcular_recompensa(current_row, action)
            total_reward += reward

            # Estado siguiente
            next_row = df_casa.iloc[i+1]
            next_state = next_row[state_columns].values.astype(np.float32)

            # Chequeo si estamos en el último step de la casa
            done = (i == len(df_casa) - 2)

            # Guardamos en memoria
            agent.remember(state, action, reward, next_state, done)

        # Realizamos replay (aprendizaje) después de cada casa
        agent.replay()

    # Reducimos epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if (episode+1) % 5 == 0:
        print(f"[Episodio {episode+1}] Recompensa total: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

print("Entrenamiento finalizado.")
