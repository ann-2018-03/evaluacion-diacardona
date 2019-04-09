# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:08:08 2019

@author: Diego Cardona
"""

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Leer el csv
df = pd.read_csv('prices_per_day.csv')
means = df['mean']


#Determinar si es necesario aplicar una transformación como logaritmo natural, raíz cúbica, raíz cuadrada, entre otros. 
plt.figure(figsize=(16,3))
plt.title('Precio_Bolsa_Nacional ($kwh)')    
plt.xlabel('Días')
plt.ylabel('kWh')
plt.plot(means, color='blue')
plt.show()



#Se evidencia en la grafica que se requiere el uso de alguna transformación, por lo tanto se hace uso de la función log
d = np.log(means)
plt.figure(figsize=(11,3))
plt.plot(d, color='black');


#Se define el modelo
class Model(object):
    def __init__(self, L):
        self.w = tf.contrib.eager.Variable([0.0] * (L))

    def __call__(self, x):
        x = tf.constant(np.array([1.0] + x, dtype=np.float32))
        y_pred = tf.reduce_sum(tf.multiply(self.w, x))
        return y_pred

    def fit(self, mu, x, y_desired):
        y_pred = self(x)
        e = y_desired - y_pred
        x = tf.constant(np.array([1.0] + x, dtype=np.float32))
        self.w.assign_add(tf.scalar_mul(2 * mu * e, x))


#Se toman los L valores previos de la serie para pronosticar el valor actual 
L = 7

##  Se crea wl modelo
model = Model(L) 

#Pronosticos del modelo (despues de probar varios mu se opta por utilizar un mu=0,0005)
y_pred = np.empty(len(d))
y_pred[:] = np.nan

for t in range(L, len(d)):
    x = d[t-L:t]
    y_pred[t] = model(x)
    model.fit(mu=0.0005, x=x, y_desired=d[t])

plt.figure(figsize=(14,3))
plt.plot(d, color='blue');
plt.plot(y_pred, color = 'red')
plt.show()

