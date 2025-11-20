import random
import numpy as np

# Criação dos dados
listAuxEntry = []
listAuxResult = []
for i in range(4):
  print("------")
  print(f"Inserindo parâmetros para linha: {i}")
  entry1 = float(input("Valor primeira entrada: "))
  entry2 = float(input("Valor segunda entrada: "))
  result1 = float(input("Valor de saída esperado: "))
  listAuxEntry.append([entry1, entry2])
  listAuxResult.append(result1)

listEntry = np.array(listAuxEntry)
listResult = np.array(listAuxResult)
listWeight = np.array([0.0, 0.0])
learningRate = 0.1

# Define se é válido entrar no passo de ativação
def next_step(value):
  return 1 if value >= 1 else 0

# Função que efetua o cálculo sobre as matrizes de entradas x pesos
def some_value(list_entry):
  return list_entry.dot(listWeight)

# Função que força o aprendizado sobre os valores enquanto estiver com erro
def learning():
  error_total = 1
  while error_total != 0:
    error_total = 0
    # Basicamente aqui vai realizar o cálculo sobre os resultados x entradas
    for i in range(len(listResult)):
      value = some_value(listEntry[i])
      result_calc = next_step(value)
      error = abs(listResult[i] - result_calc)
      error_total += error

      # Aprimorando o peso pela taxa de aprendizagem a cada loop
      for j in range(len(listWeight)):
        listWeight[j] = listWeight[j] + (learningRate * listEntry[i][j] * error)
        print(f"Peso atualizado: {listWeight[j]}")
      
    print(f"Total de erros: {error_total}")

# Executa os cálculos e retorna os produtos
learning()
print("Treinamento do modelo efetuado")
print("Resultados")
for i in range(len(listEntry)):
  print(f"Saída: {next_step(some_value(listEntry[i]))}")
print(f"Pesos finais: {listWeight}")