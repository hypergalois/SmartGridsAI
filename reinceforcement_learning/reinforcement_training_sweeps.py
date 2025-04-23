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

EPSILON_DECAY = 0.99  # Decaimiento de epsilon

import wandb
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
            'min': 0.5,
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
        'memory_size': {
            'value': 10000
        },
        'num_layers': {
            'values': [1, 2, 3, 4, 5]
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}


def sweep_train():
    wandb.init(project="SmartGrids", name="DQN-Sweep")
    config = wandb.config

    # CARGA DE DATOS Y PREPROCESADO
    dataset_consumo = pd.read_csv('datasets/dataset_consumo.csv')
    dataset_produccion = pd.read_csv('datasets/dataset_produccion.csv')
    df = pd.merge(dataset_consumo, dataset_produccion, on=['datetime','id_casa'], how='inner')
    df.sort_values(['id_casa', 'datetime'], inplace=True, ignore_index=True)

    casa_id_objetivo = 3234
    df_casa = df[df['id_casa'] == casa_id_objetivo].copy()
    df_casa.sort_values('datetime', inplace=True, ignore_index=True)

    state_columns = ['consumo_kWh', 'produccion_kWh', 'coste_euros']
    STATE_SIZE = len(state_columns)
    ACTION_SIZE = 3
    NUM_EPISODES = 20
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1

    class DQN(nn.Module):
        def __init__(self, state_size, action_size, hidden_size, num_layers):
            super().__init__()
            layers = []

            # Capa de entrada
            layers.append(nn.Linear(state_size, hidden_size))
            layers.append(nn.LeakyReLU(0.01))

            # Capas ocultas intermedias
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LeakyReLU(0.01))

            # Capa de salida
            layers.append(nn.Linear(hidden_size, action_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=config.memory_size)

            self.model = DQN(state_size, action_size, config.hidden_size, config.num_layers)
            self.target_model = DQN(state_size, action_size, config.hidden_size, config.num_layers)

            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
            self.loss_fn = nn.MSELoss()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.model.to(self.device)
            self.target_model.to(self.device)
            self.update_target_network()  # inicializar target con los mismos pesos

            self.update_counter = 0  # para saber cuándo sincronizar

        def update_target_network(self):
            self.target_model.load_state_dict(self.model.state_dict())

        def act(self, state, epsilon):
            if np.random.rand() <= epsilon:
                return random.randrange(self.action_size)
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

            # --- PREDICCIONES ---
            current_Q = self.model(states).gather(1, actions).squeeze()

            # --- TARGET con red target_model ---
            next_Q = self.target_model(next_states).max(1)[0]
            target_Q = rewards + config.gamma * next_Q * (~dones)

            # --- ENTRENAMIENTO ---
            loss = self.loss_fn(current_Q, target_Q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- ACTUALIZACIÓN DE LA RED TARGET cada N updates ---
            self.update_counter += 1
            if self.update_counter % 50 == 0:
                self.update_target_network()

            return loss.item()


    def calcular_recompensa(row, action):
        precio = row['coste_euros']
        consumo = row['consumo_kWh']
        produccion = row['produccion_kWh']
        neto = produccion - consumo
        if action == 0:
            return -abs(neto) * 0.1
        elif action == 1:
            return -1.0 if neto > 0 else 50 / precio
        else:
            return -1.0 if neto < 0 else precio / 50.0

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        total_reward = 0.0
        for i in range(len(df_casa) - 1):
            current_row = df_casa.iloc[i]
            next_row = df_casa.iloc[i + 1]
            state = current_row[state_columns].values.astype(np.float32)
            next_state = next_row[state_columns].values.astype(np.float32)
            action = agent.act(state, epsilon)
            reward = calcular_recompensa(current_row, action)
            done = (i == len(df_casa) - 2)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            loss = agent.replay()
            if loss:
                wandb.log({"loss": loss})

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        wandb.log({
            "episode": episode + 1,
            "total_reward": total_reward,
            "epsilon": epsilon
        })

    # === GUARDAR MODELO LOCAL Y EN W&B ===
    model_path = f"dqn_model_{wandb.run.name}.pth"
    torch.save(agent.model.state_dict(), model_path)

    # Subir como artifact a wandb
    artifact = wandb.Artifact(name=f"model-{wandb.run.name}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

# ==============================================
# INICIAR EL SWEEP DIRECTAMENTE DESDE EL SCRIPT
# ==============================================
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="SmartGrids")
    wandb.agent(sweep_id, function=sweep_train, count=10)