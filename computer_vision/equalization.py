import numpy as np
import pandas as pd
import cv2 as cv # Funcionalidades de visão computacional
from google.colab.patches import cv2_imshow # Imagems no colab
from skimage import io # Aplicações de processamento de imagem
from PIL import Image # Abertura e gravação de muitos formatos de imagens
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

print('='*100)
print('imagem com baixo contraste')
print('='*100)
img_low_contrast_BGR = cv.imread('baixo_contraste.jpg')
img_low_contrast_GRAY = cv.cvtColor(img_low_contrast_BGR, cv.COLOR_BGR2GRAY)
cv2_imshow(img_low_contrast_GRAY)

print('='*100)
print('imagem equalizada')
print('='*100)
img_equalized = cv.equalizeHist(img_low_contrast_GRAY)
cv2_imshow(img_equalized)

# Histograma da imagem
print('='*100)
print('Histograma da imagem com baixo contraste')
print('='*100)
plt.hist(img_low_contrast_GRAY.ravel(), bins=256, range=[0,256])
plt.show()

print('='*100)
print('Histograma da imagem equalizada')
print('='*100)
plt.hist(img_equalized.ravel(), bins=256, range=[0,256])
plt.show()