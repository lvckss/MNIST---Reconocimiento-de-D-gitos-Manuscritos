from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras

archivo = r'./imagen/tres.png'
imagen = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)

plt.imshow(imagen, cmap='gray')
plt.show()

imagen = cv2.resize(imagen, (28,28), interpolation=cv2.INTER_LINEAR)
imagen = cv2.bitwise_not(imagen)
plt.imshow(imagen, cmap='gray')
plt.show()

imagen = imagen.reshape(1, 28, 28, 1)

modelo = keras.saving.load_model('MNIST_convolucional.keras')

a = modelo.predict(imagen)

num = np.where(a == 1)[1][0]

print(num)
