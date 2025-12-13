import numpy as np
import random
import matplotlib.pyplot as plt

# Regras do jogo
  # Robô está no corredor de 5 posições [0,1,2,3,4]
  # Estado 4: Recompensa de +10
  # Outros estados: -1 de penalização
  # Robô pode se movimentar para direita +1 ou esquerda -1

# Configuração do ambiente
n_states = 5  # Número de estados
goal_state = 4 # Estado objetivo
gamma = 1.0 # Fator de desconto
episodes = 5000 # Número de episódios
max_steps = 10 # Número máximo de passos por episódio
values = np.zeros(n_states) # Inicialização dos valores dos estados
returns = {s: [] for s in range(n_states)} # Inicialização dos retornos
episodes_rewars = {s: [] for s in range(n_states)} # Inicialização das recompensas

# Geração dos episódios
def generate_episode():
  state = 0 # Sempre na posição 0
  episode = [] # Lista vazia para o episódio

  for _ in range(max_steps):
    action = random.choice([-1, 1]) # Esquerda x Direita
    next_state = state + action

    if 0 <= next_state < n_states: # Garante que se mantenha no corredor
      state = next_state
    
    reward = 10 if state == goal_state else -1 # Recompensa se chegou

    episode.append((state, reward)) # Historico do episodio

    if state == goal_state: # Trava para se chegou no objetivo
      break

  return episode


# Algoritmo monte carlo
for episode_num in range(episodes):
  episode = generate_episode() # Geração do episódio

  G = 0 # Inicialização do retorno
  visited = set() # Guarda os estados atualizados nesse episodio

  for t in reversed(range(len(episode))):
    state, reward = episode[t]
    G = gamma * G + reward # Atualiza o retorno acumulado

    if state not in visited: # Se o estado não foi visitado
      returns[state].append(G) # Adiciona o retorno ao estado
      values[state] = np.mean(returns[state]) # Atualiza o valor do estado
      visited.add(state) # marca o estado como visitado

  for s in range(n_states): # Armazena o valor atual do estado para visualização
    episodes_rewars[s].append(values[s])
      

# Exibição dos valores dos estados
print("Valores dos estados após monte carlo:")
for s in range(n_states):
  print(f"Estado {s}: Valor = {values[s]:.2f}")

# Visualização gráfica
plt.figure(figsize=(8, 4))
plt.bar(range(n_states), values, color='blue')
plt.xlabel('Estado')
plt.ylabel('Valor')
plt.title('Valores dos Estados com Monte Carlo')
plt.xticks(range(n_states))
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()