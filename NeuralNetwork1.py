"Codigo empleado para crear una red neuronal simple que convierta de grados Celsius a Farenheit"
"Obtenido de: https://www.youtube.com/watch?v=iX_on3VxZzk "
from tabnanny import verbose
import tensorflow as tf
import numpy as np

celsius= np.array([-40, -10, 0, 8, 15, 22, 38],dtype=float)
farenheit= np.array([-40, 14, 32, 46, 59, 72, 100],dtype=float)
#con mas capas
#oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
#oculta2 = tf.keras.layers.Dense(units=3)
#salida = tf.keras.layers.Dense(units=1)
#modelo = tf.keras.Sequential([oculta1, oculta2, salida])
###
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
  optimizer=tf.keras.optimizers.Adam(0.1),
  loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial=modelo.fit(celsius, farenheit, epochs=1000, verbose=False)
print("Modelo entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
#plt.show() #linea usada para mostrar el plot

print("Hagamos una predicción con 100 C!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

#pesos del modelo
print("Variables internas del modelo")
print(capa.get_weights())