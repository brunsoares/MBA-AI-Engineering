import random

# Criação dos dados
entry1 = float(input("Valor primeira entrada: "))
entry2 = float(input("Valor segunda entrada: "))
entry3 = float(input("Valor terceira entrada: "))
weight1 = random.random()
weight2 = random.random()
weight3 = random.random()
listEntry = [entry1, entry2, entry3]
listWeight = [weight1, weight2, weight3]

# Calcula o valor Entrada x Peso
def calculation(list_entry, list_weight):
  result = 0
  for i in range(len(list_entry)):
    result += list_entry[i] * list_weight[i]
  return result

# Define se é válido entrar no passo de ativação
def next_step(value):
  return value >= 1

# Executa os cálculos e retorna os produtos
result = calculation(listEntry, listWeight)
goNextStep = next_step(result)
print(f"Resultado: {result}")
print(f"Próximo passo: {goNextStep}")