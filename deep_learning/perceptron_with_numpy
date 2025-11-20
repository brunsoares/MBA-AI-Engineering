import random
import numpy as np

# Criação dos dados
entry1 = float(input("Valor primeira entrada: "))
entry2 = float(input("Valor segunda entrada: "))
entry3 = float(input("Valor terceira entrada: "))
weight1 = random.random()
weight2 = random.random()
weight3 = random.random()
listEntry = np.array([entry1, entry2, entry3])
listWeight = np.array([weight1, weight2, weight3])

# Define se é válido entrar no passo de ativação
def next_step(value):
  return value >= 1

# Executa os cálculos e retorna os produtos
result = np.dot(listEntry, listWeight)
goNextStep = next_step(result)
print(f"Resultado: {result}")
print(f"Próximo passo: {goNextStep}")