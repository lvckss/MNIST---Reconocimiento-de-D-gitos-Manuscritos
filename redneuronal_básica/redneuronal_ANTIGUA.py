"""
Código de una red neuronal simple usando solo Numpy para el dataset MNIST.
La idea de este código es afianzar los fundamentos de redes neuronales sin librerías.
Sin embargo, para esta tarea nos es mucho más útil usar una red neuronal convolucional
por lo que este código no va a ser usado en el proyecto, es simplemente una muestra de como
se aplican los fundamentos teóricos en la práctica. (red neuronal con 93% de accuracy)
"""

from redneuronal_básica.data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

imagenes, etiquetas = get_mnist()

# generamos los pesos y los sesgos de forma aleatoria en estructuras matriciales
pesos_input_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
pesos_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))
bias_input_hidden = np.zeros((20, 1))
bias_hidden_output = np.zeros((10, 1))

tasa_aprendizaje = 0.01
num_correctos = 0
iteraciones = 3
for iteracion in range(iteraciones):
    for imagen, etiqueta in zip(imagenes, etiquetas):
        imagen.shape += (1,)
        etiqueta.shape += (1,)
        # Propagación de input a hidden
        # el operador '@' realiza la multiplicación matricial gracias a la librería NumPy
        hidden_propagacion = pesos_input_hidden @ imagen + bias_input_hidden
        # realizamos la activación de la capa con la función sigmoide
        hidden_activacion = 1 / (1 + np.exp(-hidden_propagacion))
        # Propagación de hidden a output
        output_propagacion = pesos_hidden_output @ hidden_activacion + bias_hidden_output
        output_activacion = 1 / (1 + np.exp(-output_propagacion))

        # Función de costo
        # Usamos la fórmula del MSE (mean squared error o promedio del error cuadrado).
        mse = 1 / len(output_activacion) * np.sum((output_activacion - etiqueta) ** 2, axis=0)
        num_correctos += int(np.argmax(output_activacion) == np.argmax(etiqueta))

        # Retropropagación de output a hidden (derivada de la función de costo)
        derivada_output = output_activacion - etiqueta
        pesos_hidden_output += derivada_output @ hidden_activacion.T * -tasa_aprendizaje
        bias_hidden_output += derivada_output * -tasa_aprendizaje
        # Retropropagación de hidden a input (derivada de la función de activación)
        derivada_hidden = pesos_hidden_output.T @ derivada_output * (hidden_activacion * (1 - hidden_activacion))
        pesos_input_hidden += derivada_hidden @ imagen.T * -tasa_aprendizaje
        bias_input_hidden += derivada_hidden * -tasa_aprendizaje

    # Añadimos que se imprima la precision de esta iteración para monitorear como se va entrenando el modelo
    print(f"Precisión: {(num_correctos/imagenes.shape[0]) * 100}%")
    num_correctos = 0

# Guardamos los pesos y sesgos para tener el modelo ya entrenado
np.savez('modelo_entrenado.npz', pesos_input_hidden=pesos_input_hidden, pesos_hidden_output=pesos_hidden_output, bias_input_hidden=bias_input_hidden, bias_hidden_output=bias_hidden_output)

# sección de código para probar como de bien (o mal) funciona el modelo
while True:
    indice = int(input("Introduce un número del 0 al 59999: "))
    imagen = imagenes[indice]
    # usando la librería matplotlib con esta línea de código se nos abrira una ventana mostrando el número escrito a mano
    # que el modelo va a intentar predecir
    plt.imshow(imagen.reshape(28,28), cmap="Greys")

    # Realizamos la propagación con el modelo pero sin que sepa que etiqueta real tiene el input
    imagen.shape += (1,)
    # Propagación de input a hidden
    hidden_propagacion = pesos_input_hidden @ imagen.reshape(784, 1) + bias_input_hidden
    hidden_activacion = 1 / (1 + np.exp(-hidden_propagacion))
    # Propagación de hidden a output
    output_propagacion = pesos_hidden_output @ hidden_activacion + bias_hidden_output
    output_activacion = 1 / (1 + np.exp(-output_propagacion))

    plt.title(f"Valor predecido por el modelo: {output_activacion.argmax()}")
    plt.show()