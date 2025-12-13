import random
import matplotlib.pyplot as plt

# Definindo as configurações
options = {
    "Restaurante A": lambda: random.randint(8, 12), # Recompensa estável entre 8 a 12
    "Restaurante B": lambda: random.choice([0, 100]), # Recompensa mais arriscada sendo 0 ou 100
    "Restaurante C": lambda: random.randint(5, 15), # Recompensa variada entre 5 a 15
}

# Definindo parâmetros
epsilion = 0.1 # 10% de chance de explorar uma nova opção
rounds = 100 # Agente vai decidir durante 100 rodadas

# Armazenamento dos resultados
hist_points = [] # Histórico de recompensas
hist_choices = [] # Histórico de escolhas

# Valores médios estimados e número de escolhas
values = {k: 0 for k in options}
choices = {k: 0 for k in options}

# Tomada de decisão do agente
for round in range(1, rounds + 1):
  if random.random() < epsilion: # Explora nova opção
    choice = random.choice(list(options.keys()))
    c_type = "Exploração"
  else: # Vai na mais estável
    choice = max(values, key=values.get)
    c_type = "Explotação"
  
  # Guarda informações da recompensa e escolha
  reward = options[choice]() # Executa a pontuação de cada escolha
  choices[choice] += 1 # Marcação da escolha
  values[choice] += (reward - values[choice]) / choices[choice] # Média incremental

  # Guarda histórico
  hist_points.append(reward)
  hist_choices.append((round, choice, reward, c_type))

# Visualização resumida
print("TOP 10 rodadas:")
for round, choice, reward, c_type in hist_choices[:10]:
  print(f"Rodada {round}: Escolha {choice} com recompensa {reward} ({c_type})")

# Visualização completa do agente
plt.figure(figsize=(12, 6))
plt.plot(hist_points, marker='o')
plt.xlabel('Rodada')
plt.ylabel('Recompensa')
plt.title('Evolução da Recompensa ao Longo das Rodadas')
plt.grid(True)
plt.show()

# Resumo final das opções
print("Resumo das opções:")
for option in options:
  print(f"{option}: escolhido {choices[option]}x | Recompensa média estimada: {values[option]:.2f}")