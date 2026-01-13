import numpy as np
import pandas as pd
import cv2 as cv # Funcionalidades de visão computacional
from google.colab.patches import cv2_imshow # Imagems no colab
from skimage import io # Aplicações de processamento de imagem
from PIL import Image # Abertura e gravação de muitos formatos de imagens
import matplotlib.pyplot as plt

# Histogramas
print('='*100)
print('Carregar e exibir uma imagem')
print('='*100)
url = 'https://medias.lojaslinna.com.br/general/1184062_1_fullsize.jpg?w=1000&h=900&fit=fit&crop=center'
imgBGR = io.imread(url)
imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)
cv2_imshow(imgRGB)

print('='*100)
print('Histogramas individuais')
print('='*100)
# Canal Vermelho
hist = cv.calcHist([imgRGB],[0],None,[256],[0,256])
plt.title('Histograma do canal R')
plt.plot(hist, color='r')
plt.xlim([0,256])
plt.show()

# Canal Verde
hist = cv.calcHist([imgRGB],[1],None,[256],[0,256])
plt.title('Histograma do canal G')
plt.plot(hist, color='g')
plt.xlim([0,256])
plt.show()

# Canal Azul
hist = cv.calcHist([imgRGB],[2],None,[256],[0,256])
plt.title('Histograma do canal B')
plt.plot(hist, color='b')
plt.xlim([0,256])
plt.show()

print('='*100)
print('Histogramas em conjunto')
print('='*100)

color = ('r', 'g', 'b')
for i, col in enumerate(color):
  hist = cv.calcHist([imgRGB],[i],None,[256],[0,256])
  plt.plot(hist, color=col)
  plt.xlim([0,256])
plt.show()

# Exercício - Histograma
print('='*100)
print('(1) Carregar e exibir uma imagem')
print('='*100)
urlEx001 = 'https://images6.alphacoders.com/791/thumb-1920-791100.jpg'
imgBGREx001 = io.imread(urlEx001)
imgRGBEx001 = cv.cvtColor(imgBGREx001, cv.COLOR_BGR2RGB)
cv2_imshow(imgRGBEx001)

print('='*100)
print('(2) Histogramas em conjunto')
print('='*100)
for i, color in enumerate(color):
  hist = cv.calcHist([imgRGBEx001],[i],None,[256],[0,256])
  plt.plot(hist, color=color)
  plt.xlim([0,256])
plt.show()

# Converter modelos de cores
from google.colab import files
uploaded = files.upload() # Busca de arquivos no computador

print('='*100)
print('Carregar e exibir uma imagem em BGR')
print('='*100)
imgBGR = cv.imread('frutas.jpg') # Usando o CV2 no lugar do OPENCV
cv2_imshow(imgBGR)

print('='*100)
print('Converter de BGR para RGB - Errado, pois o CV necessita da imagem em BGR')
print('='*100)
imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)
cv2_imshow(imgRGB)

print('='*100)
print('Converter de BGR para GRAY')
print('='*100)
imgGRAY = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
cv2_imshow(imgGRAY)

# Exercicio - Conversão dos modelos de cores
print('='*100)
print('(1) Carregar e exibir uma imagem')
print('='*100)
uploaded = files.upload()
imgBGREx002 = cv.imread('capy.png')
cv2_imshow(imgBGREx002)

print('='*100)
print('(2) Converter de BGR para GRAY')
print('='*100)
imgGRAYEx002 = cv.cvtColor(imgBGREx002, cv.COLOR_BGR2GRAY)
cv2_imshow(imgGRAYEx002)

# Gerando histograma para auxiliar no threshold
print('='*100)
print('Gerar histograma para auxiliar na binarização')
print('='*100)
# ravel() - transforma a matriz de intensidades que tem 2D para achatada
# bins - quantidade de níveis de cinza a item tem
# range - intervalo dos valores de intensidade de cinza
plt.hist(imgGRAY.ravel(), bins=256, range=[0,256])
plt.show()

# Binarização da imagem
print('='*100)
print('Converter uma imagem de GRAY para BW (black/white - Binario)')
print('='*100)
# Return: thresh receberá o threshold - sendo 127 manualmente colocado; BW será a imagem em binário
# Parâmetros:
  # imgGRAY - Imagem em tons de cinza
  # 127 - Threshold definido, abaixo de 127 será preto (0)
  # 255 - Intensidade do pixel deve ir caso esteja acima do threshold, será branco (1)
  # THRESH_BINARY - Método para conversão do threshold
(thresh, imgBW) = cv.threshold(imgGRAY, 127, 255, cv.THRESH_BINARY)
cv2_imshow(imgBW)

# Binárização pelo metódo de otsu
print('='*100)
print('Converter uma imagem de GRAY para BW (black/white - Binario) pelo método de Otsu')
print('='*100)
(thresh, imgBW) = cv.threshold(imgGRAY, thresh, 255, cv.THRESH_OTSU)
print(f'O valor de thresh: {thresh}')
cv2_imshow(imgBW)

# Exercicio - Converter de GRAY para Binário
print('='*100)
print('(1) exibir a imagem')
print('='*100)
cv2_imshow(imgGRAYEx002)

print('='*100)
print('(2) Gerando histograma da imagem cinza')
print('='*100)
plt.hist(imgGRAYEx002.ravel(), bins=256, range=[0,256])
plt.show()

print('='*100)
print('(3) Converter de GRAY para BW (black/white - Binario)')
print('='*100)
(thresh, imgBWEx002) = cv.threshold(imgGRAYEx002, 100, 255, cv.THRESH_BINARY)
cv2_imshow(imgBWEx002)

print('='*100)
print('(4) Converter de GRAY para BW (black/white - Binario) pelo método de Otsu')
print('='*100)
(thresh, imgBWEx002) = cv.threshold(imgGRAYEx002, thresh, 255, cv.THRESH_OTSU)
print(f'O valor de thresh: {thresh}')
cv2_imshow(imgBWEx002)