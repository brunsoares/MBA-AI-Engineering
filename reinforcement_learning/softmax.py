import numpy as np
import matplotlib.pyplot as plt

# Definindo dados
review_aux = []
restaurants_aux = []
for i in range(3):
  print(f"Restaurante {i+1}")
  name = input("Digite o nome do restaurante:")
  review = float(input("Digite a nota do restaurante:"))
  restaurants_aux.append(name)
  review_aux.append(review)
  print("===========================")

reviews = np.array(review_aux)
restaurants = restaurants_aux

# Função Softmax - Notas em probabilidade
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Calculando as probabilidades
probabilities = softmax(reviews)

# Visualizando as probabilidades - Resumo
print("Probabilidades de avaliação:")
for restaurant, probability, review in zip(restaurants, probabilities, reviews):
    print(f"{restaurant} [{review}]: {probability:.2%}")

# Visualização gráfica das probabilidades
plt.figure(figsize=(8, 5))
plt.bar(restaurants, probabilities, color=['green', 'orange', 'red'])
plt.xlabel("Restaurantes")
plt.ylabel("Probabilidade")
plt.title("Probabilidades de Avaliação")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()