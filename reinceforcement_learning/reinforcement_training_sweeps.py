import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from tqdm import trange
import wandb
import sys


# Sweep config
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'total_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'gamma': {'min': 0.5, 'max': 0.999},
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
        'batch_size': {'values': [32, 64, 128]},
        'hidden_size': {'values': [64, 128, 256]},
        'memory_size': {'value': 10000},
        'num_layers': {'values': [1, 2, 3, 4, 5]}
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    }
}

def sweep_train():
    wandb.init(project="SmartGrids")
    config = wandb.config

    # ================== Carga de datos ====================
    dataset_consumo = pd.read_csv('datasets/dataset_consumo.csv')
    dataset_produccion = pd.read_csv('datasets/dataset_produccion.csv')
    df = pd.merge(dataset_consumo, dataset_produccion, on=['datetime', 'id_casa'], how='inner')
    df.sort_values(['id_casa', 'datetime'], inplace=True, ignore_index=True)
    df_casa = df[df['id_casa'] == 3234].copy().reset_index(drop=True)

    state_columns = ['consumo_kWh', 'produccion_kWh', 'coste_euros']
    STATE_SIZE = len(state_columns) + 1  # +1 por battery_soc
    ACTION_SIZE = 3
    NUM_EPISODES = 75
    EPSILON_START = 1.0
    EPSILON_MIN = 0.1
    EPSILON_DECAY = 0.98

    # ================== Red neuronal ====================
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

    # ================== Agente ====================
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

    # ================== Lógica de batería ====================
    def calcular_recompensa(row, action, battery_soc):
        precio = row['coste_euros']
        consumo = row['consumo_kWh']
        produccion = row['produccion_kWh']
        
        capacidad = 13.5  # Capacidad máxima batería (kWh)
        max_potencia = 5.0  # Potencia máxima de carga/descarga (kW)
        eficiencia = 0.95  # Eficiencia de carga/descarga
        beneficio = 0

        # Balance neto de energía
        energia_neta = produccion - consumo

        if action == 1:  # Cargar batería (comprar si hay déficit o energía barata)
            energia_a_cargar = min(max_potencia, capacidad - battery_soc)
            if energia_a_cargar > 0:
                battery_soc += energia_a_cargar * eficiencia
                coste = energia_a_cargar * precio
                beneficio = -coste

        elif action == 2:  # Descargar batería (vender si hay excedente o precio alto)
            energia_a_descargar = min(max_potencia, battery_soc)
            if energia_a_descargar > 0:
                battery_soc -= energia_a_descargar / eficiencia
                ingreso = energia_a_descargar * precio
                beneficio = ingreso

        elif action == 0:  # Usar batería para autoconsumo
            # Si hay déficit y batería disponible, usar batería
            if energia_neta < 0:
                energia_necesaria = abs(energia_neta)
                energia_de_bateria = min(energia_necesaria, battery_soc, max_potencia)
                if energia_de_bateria > 0:
                    battery_soc -= energia_de_bateria / eficiencia
                    ahorro = energia_de_bateria * precio
                    beneficio = ahorro  # Se ahorra lo que no se compra

            # Si hay excedente y batería tiene espacio, almacenar energía gratis
            elif energia_neta > 0:
                energia_almacenable = min(energia_neta, capacidad - battery_soc, max_potencia)
                if energia_almacenable > 0:
                    battery_soc += energia_almacenable * eficiencia
                    beneficio = 0  # Energía gratuita aprovechada

        return beneficio, battery_soc, beneficio

    # ================== Entrenamiento ====================
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
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
                wandb.log({"loss": loss, "beneficio": beneficio})

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        wandb.log({
            "episode": episode + 1,
            "total_reward": total_reward,
            "beneficio_euros": beneficio_total,
            "epsilon": epsilon
        })

    # Guardar modelo y subirlo a W&B
    model_path = f"dqn_model_{wandb.run.name}.pth"
    torch.save(agent.model.state_dict(), model_path)
    artifact = wandb.Artifact(name=f"model-{wandb.run.name}", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

# ============== Lanzar Sweep o Run Directo ===============
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    modo = sys.argv[1] if len(sys.argv) > 1 else "sweep"

    if modo == "sweep":
        #sweep_id = wandb.sweep(sweep_config, project="SmartGrids")
        sweep_id = "mpelaezdelarocha-u-tad/SmartGrids/d6uizf9l"
        wandb.agent(sweep_id, function=sweep_train, count=10)

    elif modo == "single":
        wandb.init(project="SmartGrids", name="agente_eléctrico", config={
            "gamma": 0.9,
            "lr": 0.00005,
            "batch_size": 24,
            "hidden_size": 64,
            "num_layers": 2,
            "memory_size": 10000,
            "epsilon_decay": 0.98
        })
        sweep_train()

    else:
        print("Modo no reconocido. Usa 'sweep' o 'single'.")

