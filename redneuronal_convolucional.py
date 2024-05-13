import numpy as np
import matplotlib.pyplot as plt

# gracias a la librería keras podemos importar el dataset de MNIST de una forma más rápida y sencilla
from keras.datasets import mnist
from keras import layers
from keras import models
from keras.utils import to_categorical

(imagenes_entrenamiento, etiquetas_entrenamiento), (imagenes_test, etiquetas_test) = mnist.load_data()

imagenes_entrenamiento = imagenes_entrenamiento.reshape((60000,28,28,1))
imagenes_entrenamiento = imagenes_entrenamiento.astype('float32')/255

imagenes_test = imagenes_test.reshape((10000,28,28,1))
imagenes_test = imagenes_test.astype('float32')/255

etiquetas_entrenamiento = to_categorical(etiquetas_entrenamiento)
etiquetas_test = to_categorical(etiquetas_test)

# Gracias a keras ahora podemos crear una red neuronal convolucional de forma mucho más sencilla y eficiente
modelo = models.Sequential()
# Añadimos diferentes capas de manejor convolucional
# Usamos como función de activación ReLU
modelo.add(layers.Conv2D(32,(3,3), activation='relu', input_shape = (28,28,1)))
modelo.add(layers.MaxPooling2D((2,2)))
modelo.add(layers.Conv2D(64,(3,3), activation='relu'))
modelo.add(layers.MaxPooling2D((2,2)))
modelo.add(layers.Conv2D(64,(3,3), activation='relu'))
modelo.add(layers.Flatten())
modelo.add(layers.Dense(64,activation = 'relu'))
# Para el output usamos al función de activación softmax que nos da unos valores entre 0 y 1 (One-hot activation entre 0 y 9)
modelo.add(layers.Dense(10, activation= 'softmax'))

# Calculamos la función de costo, en este caso con entropía cruzada
modelo.compile(optimizer = 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=5, batch_size = 64)

modelo.save('MNIST_convolucional.keras')