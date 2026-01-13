import numpy as np
import pandas as pd
import cv2 as cv # Funcionalidades de visão computacional
from google.colab.patches import cv2_imshow # Imagems no colab
from skimage import io # Aplicações de processamento de imagem
from PIL import Image # Abertura e gravação de muitos formatos de imagens
import matplotlib.pyplot as plt
from google.colab import files

# Filtros passa-baixa
uploaded = files.upload()

print('='*100)
print('Original')
print('='*100)
img_input = cv.imread('Lena_ruido.bmp')
img_input = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
cv2_imshow(img_input)

print('='*100)
print('Filtro passa-baixa 5x5')
print('='*100)
kernel = np.ones((5,5),np.float32)/25 # Máscara de 5x5 (média)
img_output = cv.filter2D(img_input,-1,kernel) # Aplica a mascara
cv2_imshow(img_output)

print('='*100)
print('Filtro passa-baixa 5x5 - Blur')
print('='*100)
blur = cv.blur(img_input,(5,5)) # Aplica a mascara
cv2_imshow(blur)

print('='*100)
print('Filtro passa-baixa 5x5 - Gaussian Blur')
print('='*100)
gauss = cv.GaussianBlur(img_input,(5,5),0) # Aplica a mascara
cv2_imshow(gauss)

print('='*100)
print('Filtro passa-baixa 5x5 - Median Blur')
print('='*100)
median = cv.medianBlur(img_input,5) # Aplica a mascara
cv2_imshow(median)

# Filtros passa-alta
# uploaded = files.upload()

print('='*100)
print('Original')
print('='*100)
img_input = cv.imread('Moon.tif')
img_input = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
cv2_imshow(img_input)

print('='*100)
print('Filtro gradiente manual')
print('='*100)
kernel = np.array([[1,  1, 1],
                   [0,  0, 0],
                   [-1,-1,-1]], np.float32)
img_output = cv.filter2D(img_input,-1,kernel) # Aplica a mascara
cv2_imshow(img_output)

print('='*100)
print('Filtro gradiente manual')
print('='*100)
kernel = np.array([[ 5,  5,  5,  5,  5,  5],
                   [ 3,  3,  3,  3,  3,  3],
                   [ 0,  0,  0,  0,  0,  0],
                   [-3, -3, -3, -3, -3, -3],
                   [-5, -5, -5, -5, -5, -5]], np.float32)
img_output = cv.filter2D(img_input,-1,kernel) # Aplica a mascara
cv2_imshow(img_output)

print('='*100)
print('Filtro passa-alta')
print('='*100)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], np.float32)
img_output = cv.filter2D(img_input,-1,kernel) # Aplica a mascara
cv2_imshow(img_output)

print('='*100)
print('Filtro laplaciano')
print('='*100)
kernel = np.array([[ 1,  1,  1],
                   [ 1, -8,  1],
                   [ 1,  1,  1]], np.float32)
img_output = cv.filter2D(img_input,-1,kernel) # Aplica a mascara
cv2_imshow(img_output)

print('='*100)
print('Filtro derivativos')
print('='*100)
img_input = cv.imread('Moon.tif')
img_input = cv.cvtColor(img_input, cv.COLOR_BGR2GRAY)
laplacian = cv.Laplacian(img_input,cv.CV_64F)
sobelx = cv.Sobel(img_input,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img_input,cv.CV_64F,0,1,ksize=5)
print('Original')
cv2_imshow(img_input)
print('Laplacian')
cv2_imshow(laplacian)
print('Sobel X')
cv2_imshow(sobelx)
print('Sobel Y')
cv2_imshow(sobely)

print('='*100)
print('Filtro passa-alta canny')
print('='*100)
canny = cv.Canny(img_input,100,200)
cv2_imshow(canny)