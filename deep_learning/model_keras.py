import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Importando os dados
x = np.array([[0,0], [0,1], [1,0], [1,1]]) # Entradas
y = np.array([[0], [1], [1], [0]]) # Saídas

# Definindo rede neural com Keras
model = tf.keras.Sequential([
    # Parâmetros = Qtd de neurônios, recebe quantos neurônios, função de ativação, nome da camada
    tf.keras.layers.Dense(3, input_dim=2, activation='sigmoid', name='hidden_layer'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Visualizando a rede neural
model.summary()

# Compilando modelo
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), # Otimizador
              loss='binary_crossentropy', # Função de perde
              metrics=['binary_accuracy']) # Métricas - O quanto ele está acertando

# Treinando modelo
# Parâmetros - dados entrada, dados saída, épocas, log do processamento
history = model.fit(x, y, epochs=500, verbose=1)

# Validando resultado - Visualizando com matplotlib
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['binary_accuracy'], label='Accuracy')
plt.title('Treinamento modelo XOR com Keras')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

# Previsões do modelo
predictions = model.predict(x)
print(f"{'Entrada':<12}{'Saída':<15}{'Resposta Modelo':<15}{'Resultado'}")
for entry, expected, out in zip(x, y, predictions):
  entryStr = f"{entry[0]}, {entry[1]}"
  outBin = 1 if out > 0.5 else 0
  result = "✅" if outBin == expected else "❌"
  print(f"{entryStr:<12}->{expected[0]:<15}->{outBin:<15}{result}")

# Visualizando os pesos
model.get_weights()

# Salvando modelo
path = 'xor_model.keras'
model.save(path)
print(f"Modelo salvo em: {path}")

# Carregando modelo salvo
modelLoaded = tf.keras.models.load_model(path)
modelLoaded.summary()