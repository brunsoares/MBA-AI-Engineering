import numpy as np

# Função de ativação sigmoid
def sigmoid(value):
  return  1 / (1 + np.exp(-value))

# Função sigmoid derivada
def sigmoid_derivate(value):
  return value * (1 - value)

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
  listAuxResult.append([result1])

listEntry = np.array(listAuxEntry)
listResult = np.array(listAuxResult)
listWeightIn = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
listWeightOut = np.array([[-0.017], [-0.893], [0.148]])
momentum = 1
learningRate = 0.3
epoch = 1000

for j in range(epoch):
  # Treinamento do modelo
  # Cálculo camada entrada - Ativação Sigmoid
  layerIn = listEntry
  resultSinapse = np.dot(layerIn, listWeightIn)
  layerHidden = sigmoid(resultSinapse)
  print("Resultados - cálculo camada entrada - Ativação Sigmoid")
  print(layerHidden)
  print("==================")

  # Cálculo camada saída - Ativação sigmoid
  resultFinal = np.dot(layerHidden, listWeightOut)
  layerOut = sigmoid(resultFinal)
  print("Resultados - cálculo camada saída - Ativação Sigmoid")
  print(layerOut)
  print("==================")

  # Cálculo do erro
  layerError = listResult - layerOut
  print("Resultados - cálculo do erro")
  print(layerError)
  print("==================")

  # Cálculo da média absoluta
  layerMean = np.mean(np.abs(layerError))
  print("Resultados - cálculo da média absoluta")
  print(layerMean)
  print("==================")

  # Cálculo da derivada da saída
  derivadaOut = sigmoid_derivate(layerOut)
  print("Resultados - cálculo da derivada da saída")
  print(derivadaOut)
  print("==================")

  # Cálculo delta - Camada saída
  deltaOut = layerError * derivadaOut
  print("Resultados - cálculo delta - Camada saída")
  print(deltaOut)
  print("==================")

  # Cálculo delta - Camada escondida - Precisamos transpor a matriz
  weightTrans = listWeightOut.T
  deltaHidden = deltaOut.dot(weightTrans)
  layerDeltaHidden = deltaHidden * sigmoid_derivate(layerHidden)
  print("Resultados - cálculo delta - Camada escondida")
  print(layerDeltaHidden)
  print("==================")

  # Cálculo do backpropagation - Encontrar minimos globais
  # Transpor a matriz de camada oculta
  layerHiddenTrans = layerHidden.T
  newWeightsOut = layerHiddenTrans.dot(deltaOut)
  listWeightOut = (listWeightOut * momentum) + (newWeightsOut * learningRate)
  print("Resultados - cálculo backpropagation")
  print(listWeightOut)
  print("==================")

  # Atualização da camada de entrada
  layerInTrans = layerIn.T
  newWeightsIn = layerInTrans.dot(layerDeltaHidden)
  listWeightIn = (listWeightIn * momentum) + (newWeightsIn * learningRate)
  print("Resultados - atualização da camada de entrada")
  print(listWeightIn)
  print("==================")

# Confirmação dos resultados
print("Confirmação dos resultados")
print(f"Pesos entradas: {listWeightIn}")
print(f"Pesos saída: {listWeightOut}")
print("==================")