import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Base de dados
# Não vamos utilizar dados de testes, somente treinamento
(xTrain, yTrain), (_,_) = tf.keras.datasets.mnist.load_data()

# Validando os dados
print(xTrain.shape)
print(yTrain.shape)

# Visualizando as imagens tendo X a imagem e Y classificação
i = np.random.randint(0, 6000)
print(f'Classificação: {yTrain[i]}')
plt.imshow(xTrain[i], cmap='gray')

# Precisamos adicionar mais uma camada que é o canal de cor (preto x branco)
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1).astype('float32')
print(xTrain.shape)

# Visualizando a faixa de valor
print(f'Menor valor: {xTrain[0].min()}, Maior valor: {xTrain[0].max()}')

# Normalizando os dados entre -1 e 1
# 127.5 = (255/2)
xTrain = (xTrain - 127.5) / 127.5

# Visualizando nova faixa de valor
print(f'Menor valor: {xTrain[0].min()}, Maior valor: {xTrain[0].max()}')

# Dividindo em mini batch gradient descent
buffer_size = 60000
batch_size = 256

# Precisamos mudar para o formato do TensorFlow e misturaremos o dataset
xTrain = tf.data.Dataset.from_tensor_slices(xTrain).shuffle(buffer_size).batch(batch_size)
print(type(xTrain))

# Construindo o Gerador
def create_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU()) # Função dea ativação 
    model.add(layers.Reshape((7, 7, 256))) # Alterando a entrada de vetor para matriz

    # 7x7x128
    # Filters é o detector de caracteristicas
    # Kernel_size é o tamanho da matriz que fará a multiplicação do filtro pela imagem
    # Paddind = same é a identificação para que todos os pixels sejam utilizados nas caracteristicas
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 14x14x64
    # Strides é de quantos x quantos pixels que os calculos são feitos
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # 28x28x1
    # Filters 1 é um único canal
    model.add(layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    print(model.summary())

    return model

generator = create_generator()

# Visualizar o formato da entrada do gerador
print(generator.input_shape)

# Gerando ruído de imagem randomica
noise = tf.random.normal([1, 100])
print(noise.shape)

# Gerando e visualizando a primeira imagem
new_image = generator(noise, training=False)
print(new_image.shape)
plt.imshow(new_image[0, :, :, 0], cmap='gray')

# Construição do discriminador
def create_discriminator():
  # Objetivo - receber as imagens e fazer classificação binária
  model = tf.keras.Sequential()

  # 14x14x64
  # Conv2D é utilizado para detectar caracteristicas
  model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))

  # 7x7x128
  model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  
  model.add(layers.Flatten())
  model.add(layers.Dense(1))

  print(model.summary())

  # Vamos deixar sem a função de ativação, assim podemos usar a sigmoid
  return model

discriminator = create_discriminator()

# Visualizando a entrada da camada do discriminador
print(discriminator.input_shape)

# Testando a imagem do ruído para verificar se é uma imagem real ou não
# Vai retornar o logits que é o resultado da rede neural sem função de ativação
# Resultado bruto da rede neural
discriminator(new_image, training=False)

# Cálculo do erro em cima do resultado bruto
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

print(tf.ones_like(1))
print(tf.zeros_like(1))

# Calculador da perda do discriminador que irá receber imagens reais x fakes
def discriminator_loss(real_output, fake_output):
  # Criando matriz de 1s e de 0s para comparar
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

# Interligação das duas redes Gerador x Discriminador
# Objetivo do gerador é criar imagens mais próximas das reais
def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output)

# Atualizando os pesos das duas redes
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# Treinamento da GAN e visualização dos resultados
epochs = 100
noise_dimension = 100
qtd_images = 16

# Função do treinamento
# Usamos o @ para melhor desempenho com tf.function
@tf.function
def training(images):
  noise = tf.random.normal([batch_size, noise_dimension])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    # Enviamos as imagens do dataset mnist e as imagens fakes
    expected_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    # Calculamos as perdas
    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(expected_output, fake_output)

  # Ajustes nos pessos de cada rede
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  # Aplicamos as atualizações - Envio da "dica"
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Visualizando imagens de teste
test_images = tf.random.normal([qtd_images, noise_dimension])
print(test_images.shape)

# Treinando a GAN
def train_gan(dataset, epochs, test_images):
  for epoch in range(epochs):
    for image_batch in dataset:
      training(image_batch)

    print(f'Época: {epoch + 1}')
    # Depois do treinamento de 1 batch, vamos visualizar os resultador por epoca
    generated_images = generator(test_images, training=False)

    # Visualizando os resultados
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
      plt.subplot(4, 4, i+1)
      # Precisamos denormalizar os resultados usando 127.5 + 127.5
      plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
    plt.show()

# Realizando o treinamento das redes
train_gan(xTrain, epochs, test_images)