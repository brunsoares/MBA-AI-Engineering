import numpy as np
import random
import matplotlib.pyplot as plt

# Regra do caminho
  # Robô começa no canto (2, 0)
  # bateria está no canto (0, 4)
  # Areas # são obstaculos e não pode passar
  # Demais areas são espaços livres
  # Pode se moder para cima, baixo, direita, esquerda
  # Recompensa:
    # +10 ao chegar na bateria
    # -1 a cada passo
    # -5 se bater no obstáculo
# Objetivo: Ensinar o melhor caminho pelo Processo de decisão de markov 

# Criando cenário
grid = [
    ['', '', '', '', 'B'],
    ['', '#', '', '#', ''],
    ['R', '', '', '', ''],
]

# Parâmetros iniciais
rows, cols = 3, 5 # Tamanho do cenário
actions = ['up', 'down', 'left', 'right']
q_table = {}

# Função de recompensa
def reward(state):
    i, j = state
    cell = grid[i][j]
    if cell == 'B': # Encontrou a bateria
        return 10
    elif cell == '#': # Bateu no obstaculo
        return -5
    else:
        return -1 # Deu um movimento
    
# Função movimento
def move(state, action):
    i, j = state
    if action == 'up':
        i = max(0, i-1)
    elif action == 'down':
        i = min(rows-1, i+1)
    elif action == 'left':
        j = max(0, j-1)
    elif action == 'right':
        j = min(cols-1, j+1)
      
    if grid[i][j] == "#": # Obstaculo não anda
      return state

    return (i, j)

# Iniciando Q-TABLE
for i in range(rows):
    for j in range(cols):
        if grid[i][j] != '#':
            for action in actions:
                q_table[(i, j)] = {a: 0 for a in actions}

# Loop de aprendizado Q-Learning
episodes = 300
alpha = 0.1
gamma = 0.9
epsilon = 0.2
rewards_per_episode = []

# Laço de aprendizado
for ep in range(episodes):
    state = (2, 0)
    total_reward = 0
    
    for _ in range(50):
      if random.random() < epsilon:
        action = random.choice(actions) # Exploração
      else:
        action = max(q_table[state], key=q_table[state].get) # Explotação

      next_state = move(state, action)
      reward_value = reward(next_state)
      total_reward += reward_value

      if next_state in q_table:
        best_next_move = max(q_table[next_state].values())
      else:
        best_next_move = 0

      q_table[state][action] += alpha * (reward_value + gamma * best_next_move - q_table[state][action])
      state = next_state

      if grid[state[0]][state[1]] == 'B': # Chegou na bateria
        break

    rewards_per_episode.append(total_reward)

# Visualização das recompensas por episodio
plt.plot(rewards_per_episode)
plt.xlabel('Episódios')
plt.ylabel('Recompensas')
plt.title('Recompensas por Episódio')
plt.grid(True)
plt.show()

# Visualizando os melhores caminhos
print("Melhor ação por célula:")
for i in range(rows):
    line = ''
    for j in range(cols):
        if grid[i][j] == '#':
          line += '  #  '
        elif grid[i][j] == 'B':
          line += '  B  '
        else:
          best_action = max(q_table[(i, j)], key=q_table[(i, j)].get)
          direction = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[best_action]
          line += f'  {direction}  '
    print(line)
