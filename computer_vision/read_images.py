import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv # Funcionalidades de visão computacional
from google.colab.patches import cv2_imshow # Imagems no colab
from skimage import io # Aplicações de processamento de imagem
from PIL import Image # Abertura e gravação de muitos formatos de imagens

# Ler e exibir imagens - URL DIGITAL
urls = [
    'https://iiif.lib.ncsu.edu/iiif/0052574/full/800,/0/default.jpg',
    'https://iiif.lib.ncsu.edu/iiif/0016007/full/800,/0/default.jpg'
]

for url in urls:
  imgBGR = io.imread(url) # Leitura da imagem, convertendo para matrix de pixel com modelo de cor BGR
  imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB) # Conversão para modelo de cor RGB
  final_frame = cv.hconcat([imgBGR, imgRGB]) # Concatenação das imagens
  cv2_imshow(final_frame) # Exibição das imagens
  print('\n')

# Ler e exibir imagens - DRIVE do usuário
from google.colab import drive
drive.mount('/content/drive')

img_drive_BGR = io.imread('/content/drive/MyDrive/Desenvolvimento/Lenna_RGB.png') # Caminho pelo diretório do drive do usuário
img_drive_RGB = cv.cvtColor(img_drive_BGR, cv.COLOR_BGR2RGB) # Converte para matriz RGB
final_frame_drive = cv.hconcat([img_drive_BGR, img_drive_RGB]) # Concatenação das imagens
cv2_imshow(final_frame_drive) # Visualização da imagem

# Ler e exibir imagens - Imagens do computador
from google.colab import files
uploaded = files.upload() # Upload das imagens

img_comp_BGR = io.imread('/content/Lenna_RGB.png') # Caminho da imagem do computador
img_comp_RGB = cv.cvtColor(img_comp_BGR, cv.COLOR_BGR2RGB) # Converte para matriz RGB
final_frame_comp = cv.hconcat([img_comp_BGR, img_comp_RGB]) # Concatenação das imagens
cv2_imshow(final_frame_comp) # Visualização da imagem

# Acessar informações da imagem
print('='*100)
print('Carregar e exibir uma imagem')
print('='*100)
url = 'https://static.wikia.nocookie.net/computervision/images/3/34/Lenna.jpg'
IBGR = io.imread(url) # Leitura da imagem, convertendo para matrix de pixel com modelo de cor BGR
IRGB = cv.cvtColor(IBGR, cv.COLOR_BGR2RGB) # Conversão para modelo de cor RGB
cv2_imshow(IRGB) # Exibição da imagem

print('='*100)
print('Informações da imagem')
print('='*100)
print(f'Tipo de dados: {str(IRGB.dtype)}')
print(f'Altura: {IRGB.shape[0]}')
print(f'Largura: {IRGB.shape[1]}')
print(f'Qtd Canais: {IRGB.shape[2]}')

print('='*100)
print('Acessar matrizes da imagem')
print('='*100)
print('Acessando uma matriz com 3 dimensões')
print(IRGB[0,0,0]) # É o valor de VERMELHO do primeiro pixel da imagem (0,0)
print(IRGB[0,0,1]) # É o valor de VERDE do primeiro pixel da imagem (0,0)
print(IRGB[0,0,2]) # É o valor de AZUL do primeiro pixel da imagem (0,0)

print('='*100)
print('Acessando uma matriz com 2 dimensões')
r = IRGB[:,:,0] # Canal VERMELHO
g = IRGB[:,:,1] # Canal VERDE
b = IRGB[:,:,2] # Canal AZUL
print(r[0,0]) # É o valor de VERMELHO do primeiro piuxel da imagem (0,0)
print(g[0,0]) # É o valor de VERDE do primeiro piuxel da imagem (0,0)
print(b[0,0]) # É o valor de AZUL do primeiro piuxel da imagem (0,0)

print('='*100)
print('Acessando a matriz inteira')
print(r)

print('='*100)
print('Acessar uma região da matriz')
print(r[0:2,0:9])

# Exercício - Acesso de matriz
uploaded = files.upload() # Upload das imagens

ex_ibgr = io.imread('/content/haikyuu.png') # Caminho da imagem do computador
ex_irgb = cv.cvtColor(ex_ibgr, cv.COLOR_BGR2RGB) # Converte para matriz RGB
print('1- Exibir imagem convertida RGB')
cv2_imshow(ex_irgb) # Visualização da imagem
print('='*100)
print('2- Exibindo dimensão da imagem')
print(f'{ex_irgb.shape[0]}x{ex_irgb.shape[1]}')
print('='*100)
print('3- Exibindo matriz verde')
g = ex_irgb[:,:,1] # Canal VERDE
print(g)
print('='*100)
print('4- Exibindo as 3 cores do último pixel')
print('='*100)
print(ex_irgb[ex_irgb.shape[0]-1,ex_irgb.shape[1]-1,0]) # -1 por conta da matriz começar no 0
print('5- Exibir valores de azul para linha central da imagem, imprimindo todas as colunas')
b = ex_irgb[ex_irgb.shape[0]//2,:,2]
print(b)
print('='*100)
print('6- Exibir valores de vermelho para coluna central da imagem, imprimindo todas as linhas')
r = ex_irgb[:,ex_irgb.shape[1]//2,0]
print(r)
