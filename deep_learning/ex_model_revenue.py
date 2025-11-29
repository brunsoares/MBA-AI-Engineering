import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Importação da base de dados
sales_df = pd.read_csv('SalesData.csv')
sales_df.reset_index(drop=True, inplace=True)
print(sales_df) # Todos os dados
print("----")
print(sales_df.head()) # Top 5
print("----")
print(sales_df.tail()) # Últimos 5
print("----")
print(sales_df.info()) # Informações
print("----")
print(sales_df.describe()) # Estatísticas

# Visualização da base de dados
sns.scatterplot(x=sales_df['Temperature'], y=sales_df['Revenue'])
plt.show()

# Configuração da base de treinamento
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']
print(x_train.shape)
print(y_train.shape)

# Configuração e treinamento do modelo
model = tf.keras.Sequential()
# Apenas uma camada - Erro fica muito longe do 0
# model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
# Múltiplas camadas
model.add(tf.keras.layers.Dense(units=10, input_shape=[1]))
model.add(tf.keras.layers.Dense(units=1))

print(model.summary()) # Já está adicionando o BIAS no parâmetros (mesmo definindo apenas 1 parâmetro)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5), loss='mean_squared_error')
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
temp_c = 21
revenue = model.predict(np.array([temp_c]))
print("Hoje será vendido por volta de R$", np.round(revenue,2))
plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Temperature vs Revenue')
plt.show()