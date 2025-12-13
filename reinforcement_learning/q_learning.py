import numpy as np
import random
import matplotlib.pyplot as plt

# Configuração do ambiente
places = ['Casa', 'Rua Movimentada', 'Sorveteria', 'Escola', 'Desvio', 'Rua Tranquila']
n_places = len(places) # Número total de lugares
actions = ['Ir para o próximo lugar', 'Ir para o lugar anterior'] # Ações possíveis 0 ou 1
rewards = {
    "Casa": 0,
    "Rua Movimentada": -1,
    "Sorveteria": 2,
    "Escola": 5,
    "Desvio": -1,
    "Rua Tranquila": 0
} # Definindo as recompensas por estado
q_table = np.zeros((n_places, len(actions))) # Inicializando a tabela Q

# Parâmetros do Q-Learning
alpha = 0.1 # Taxa de aprendizado
gamma = 0.9 # Fator de desconto
epsilon = 0.1 # Taxa de exploração
episodes = 2000 # Número de tentativas

# Função auxiliar para converter o nome dos locais em indices
def place_to_index(place):
    return places.index(place)

# Função de politica epsilon-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0,1]) # Exploração: Ação aleatória
    else:
        return np.argmax(q_table[state]) # Explotação: Melhor ação
    
# Função de gerar episodio
def generate_episode(path):
  episode = []
  for i in range(len(path) - 1):
    current_place = path[i]
    next_place = path[i + 1]
    reward = rewards[next_place]
    episode.append((current_place, reward, next_place)) # Historico do episodio
  return episode

# Possíveis caminhos
path1 = ['Casa', 'Rua Movimentada', 'Sorveteria', 'Escola']
path2 = ['Casa', 'Desvio', 'Rua Tranquila', 'Escola']

# Treinamento do agente
for episode_n in range(episodes):
    path = path1 if episode_n % 2 == 0 else path2 # Alterna entre os episodios
    episode = generate_episode(path) # Gerando o episodio

    # Começa com o primeiro estado do episodio
    current_place = episode[0][0]
    current_index = place_to_index(current_place)

    for i in range(len(episode)):
        reward = episode[i][1]
        
        if i == len(episode) - 1: # Último estado do episódio
            td_error = reward - q_table[current_index, choose_action(current_index)] # Erro temporal (TD)
            q_table[current_index, choose_action(current_index)] += alpha * td_error # Atualiza a tabela Q
            break
        
        # Próximo local e índice
        next_place = episode[i + 1][0]
        next_index = place_to_index(next_place)
        
        # Atualização do q-learning
        best_future_act = np.argmax(q_table[next_index]) # Melhor ação futura
        td_error = reward + gamma * q_table[next_index, best_future_act] - q_table[current_index, choose_action(current_index)] # Erro temporal

        q_table[current_index, choose_action(current_index)] += alpha * td_error # Atualiza a tabela Q
        current_index = next_index # Atualiza o estado atual

# Exibição dos valores Q
print("Valores Q após aprendizado Q-Learning:")
for idx, place in enumerate(places):
    # Obtem ações dos valores Q
    q_next = q_table[idx][0]
    q_back = q_table[idx][1]
    
    # Define qual é a melhor ação: Maior valor Q
    best_action = "Ir para o próximo lugar" if q_next >= q_back else "Ir para o lugar anterior" 

    # identifica o proxima e anterior
    next_place = places[idx + 1] if idx + 1 < len(places) else "Nenhum (FINAL)"
    back_place = places[idx - 1] if idx - 1 >= 0 else "Nenhum (INICIO)"

    # Exibição dos resultados
    print(f"Local: {place}")
    print(f"Valor Q: {q_next:.2f} | Próximo local: {next_place}")
    print(f"Valor Q: {q_back:.2f} | Local anterior: {back_place}")
    print(f"Ação escolhida: {best_action}")
    print("=======\n")
