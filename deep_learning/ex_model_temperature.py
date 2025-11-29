import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Importação da base de dados
temperature_df = pd.read_csv('data/Celsius-to-Fahrenheit.csv')
temperature_df.reset_index(drop=True, inplace=True)
print(temperature_df) # Todos os dados
print("----")
print(temperature_df.head()) # Top 5
print("----")
print(temperature_df.tail()) # Últimos 5
print("----")
print(temperature_df.info()) # Informações
print("----")
print(temperature_df.describe()) # Estatísticas

# Visualização da base de dados
sns.scatterplot(x=temperature_df['Celsius'], y=temperature_df['Fahrenheit'])
plt.show()

# Configuração da base de treinamento
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']
print(x_train.shape)
print(y_train.shape)

# Configuração e treinamento do modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
print(model.summary()) # Já está adicionando o BIAS no parâmetros (mesmo definindo apenas 1 parâmetro)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
epoch_hist = model.fit(x_train, y_train, epochs=1000, verbose=1)

# Avaliação do modelo
print(model.get_weights())
print(epoch_hist.history.keys())
plt.plot(epoch_hist.history['loss'])
plt.title('Progresso de perda durante treinamento do modelo')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.show()

# Predição do modelo
temp_c = -5
temp_f = model.predict(np.array([temp_c]))
print(f'Temperatura em Celsius: {temp_c}')
print(f'Temperatura em Fahrenheit: {np.round(temp_f,1)}')