import math
import random
import matplotlib.pyplot as plt

# Definindo os dados
probabilities = {
    "Máquina A": 0.3, # Paga 30% das jogadas
    "Máquina B": 0.5, # Paga 50% das jogadas
    "Máquina C": 0.7  # Paga 70% das jogadas
}

# Parâmetros da simulação
rounds = 50 # Quantidade de rodadas
victories = {m: 0 for m in probabilities} # Quantidade de vitórias por máquina
plays = {m: 0 for m in probabilities} # Quantidade de jogadas por máquina
hist_points = [] # Histórico de pontos

# Função UCB - Cálculo de confiança
def ucb(media, total_plays, x_plays):
    if x_plays == 0: # Se eu nunca joguei, preciso jogar
        return float('inf')
    # Quanto menos testada a máquina, mais bônus vai ter
    reward = math.sqrt(2 * math.log(total_plays) / x_plays)
    return media + reward

# Simulação das rodadas
for round in range(1, rounds + 1):
    values_ucb = {}

    # 1) Calcula UCB para cada máquina
    for machine, p in probabilities.items():
        # Média de vitórias até agora
        if plays[machine] > 0:
            average_points = victories[machine] / plays[machine]
        else:
            average_points = 0
        
        # Calcula o valor UCB da média + incerteza
        values_ucb[machine] = ucb(average_points, round, plays[machine])

    # 2) Escolhe a máquina com maior UCB
    choice = max(values_ucb, key=values_ucb.get)

    # 3) Simula jogada
    reward = 1 if random.random() < probabilities[choice] else 0

    # 4) Atualiza estatísticas
    plays[choice] += 1
    victories[choice] += reward
    hist_points.append(reward)

    # 5) Feedback em tempo real
    print(f"Rodada {round}: Máquina escolhida: {choice}, Pontos obtidos: {reward}, UCB: {values_ucb[choice]:.2f}")
    print("================\n")

# Visualização gráfica
earning_amount = [sum(hist_points[:i]) for i in range(len(hist_points) + 1)]
plt.figure(figsize=(10, 5))
plt.plot(earning_amount, marker='o')
plt.xlabel('Rodada')
plt.ylabel('Ganho acumulado')
plt.title('Ganho acumulado por rodada UCB')
plt.grid(True)
plt.show()

# Resumo final
print("Resumo Final:")
for machine in probabilities:
  total = plays[machine]
  victory_rate = (victories[machine] / total) * 100 if total > 0 else 0
  print(f"Máquina: {machine}, Total de Jogadas: {total}, Total de Vitórias: {victories[machine]}, Taxa de Vitória: {victory_rate:.2f}%")