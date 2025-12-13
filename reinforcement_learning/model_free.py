import numpy as np
import random
import matplotlib.pyplot as plt

# Regras do jogo
  # O agente não sabe o que vai acontecer até tentar
  # Ele não conhece o ambiente nem recompensas
  # Aprende com tentativa e erro
  # Com o tempo, ele ajusta suas ações para maximizar a recompensa

# Criação do cenário
env = np.array([
    [ 0, 0, 0, 0, 10 ], # Linha 0 - Tem a bateria (+10)
    [ 0, -1, 0, -1, 0 ], # Linha 1 - Tem obstaculos (-5 penalidade)
    [ 0, 0, 0, 0, 0 ] # Linha 2 - Ponto inicial será 2,0
])

# Parâmetros iniciais
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((3, 5, len(actions)))

# Hiperparâmetros
alpha = 0.1 # Taxa de aprendizado
gamma = 0.9 # Fator de desconto
epsilon = 0.1 # Taxa de exploração
decay_rate = 0.995 # Taxa de redução do epsilon

steps_per_episode = [] # Historico de passos que levou por episodio

# Função de escolher ação (epsilon-greedy)
def choose_action(state):
  if random.uniform(0, 1) < epsilon:
    return random.choice(actions) # exploração
  else:
    row, col = state
    return actions[np.argmax(q_table[row, col])] # Explotação
  
# Função de executar uma ação e recompensa
def execute_action(state, action):
  row, col = state
  if action == 'up':
    row = max(row - 1, 0)
  elif action == 'down':
    row = min(row + 1, 2)
  elif action == 'left':
    col = max(col - 1, 0)
  elif action == 'right':
    col = min(col + 1, 4)

  if env[row, col] == -1: # Bateu em um obstaculo, não anda e recebe -5
    return state, -5
  
  if env[row, col] == 10: # Chegou na bateria, anda e recebe +10
    return (row, col), 10
  
  return (row, col), -1 # No final, deu um passo perde -1

# Laço principal de treinamento
for episode in range(200):
  state = (2, 0) # Ponto inicial
  steps = 0

  while True:
    action = choose_action(state) # Escolhe uma ação
    next_state, reward = execute_action(state, action) # Executa essa ação
    row, col = state # coordenadas do estado atual
    action_index = actions.index(action) # índice da ação
    next_row, next_col = next_state # coordenadas do próximo estado
    best_next_action = np.max(q_table[next_row, next_col]) # melhor ação no próximo estado

    # Atualização da Q-Table para registro
    # Robô ajusta sua expectativa com o que ele acabou de vivenciar
    q_table[row, col, action_index] += alpha * (reward + gamma * best_next_action - q_table[row, col, action_index])

    # Novo estado vira estado atual (robô anda)
    state = next_state
    steps += 1

    # Chegou na bateria ou tentou por muito tempo 50x
    if env[state[0], state[1]] == 10 or steps > 50:
      break
  
  # Armazena passos do robô no episodio
  steps_per_episode.append(steps)

  # Diminui taxa de exploração
  epsilon *= decay_rate

# Visualizando o aprendizado do robô
plt.figure(figsize=(12, 5))
plt.plot(steps_per_episode, color='blue')
plt.xlabel('Episódio')
plt.ylabel('Passos')
plt.title('Aprendizado do robô')
plt.grid(True)
plt.show()

# Mostrando o aprendizado
print("Aprendizado:")
policy = np.full((3, 5), '', dtype=object)

for row in range(3):
  for col in range(5):
    if env[row,col] == -1: # Obstaculo
      policy[row, col] = '#'
    elif env[row,col] == 10: # Bateria
      policy[row, col] = 'B'
    else: # Caminho efetuado
      best_action = np.argmax(q_table[row, col])
      symbol = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
      policy[row, col] = symbol[actions[best_action]]

print(policy)
