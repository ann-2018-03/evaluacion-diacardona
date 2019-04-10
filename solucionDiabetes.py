# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:30:47 2019

@author: Diego Cardona
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.neighbors import KNeighborsRegressor


## leer de las columnas
columnName = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'y']

## Leer el archivo CSV
dbDiabetes = pd.read_csv('diabetes.csv', names = columnName)

#Determinar cuáles de las variables consideradas son relevantes para el problema.
#Se llega a la conclusión despues de leer la teoria sobre como se diagnostica la diabetes, que se utilizaran las variables
#"serum 6"  que es la cantidad de glucosa en la sangre y "Y"

X = dbDiabetes.s6.tolist()
Y = dbDiabetes.y.tolist()

#Determinar si hay alguna transformación de las variables de entrada o de salida que mejore el pronóstico del modelo.
X = [float(y) for y in X[1:]]
Y = [float(z) for z in Y[1:]]
print(X)
print(Y)

X.sort()
Y.sort()

plt.figure(0)
plt.plot(X)

plt.figure(1)
plt.plot(Y)

## Convierte todas las columnas a float
##dbDiabetes = [[float(y) for y in x] for x in dbDiabetes]

#for item in ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']:
#    X = dbDiabetes[[item]]
#    Y = dbDiabetes['Y']
#    
#    dbDiabetes.sort_values(item)[item]
#    
#    lm = LinearRegression()
#    print(item, cross_val_score(lm, X, Y, cv=20).mean())
    

# partimos del hecho que solo 2 variables son relevantes para el modelo, s6 = glucosa y 'Y' el nivel de azucar en la sangre 
#plt.figure(0)
#plt.plot(dbDiabetes.sort_values('s6')['s6'],range(len(dbDiabetes.sort_values('age')['age'])))
#
#plt.figure(2)
#plt.plot(dbDiabetes.sort_values('y')['y'],range(len(dbDiabetes.sort_values('age')['age'])))

##
## Sumatoria del error cuadrático
##
def SSE(w0, w1):
    return (sum( [(v - w0 - w1*u)**2  for u, v in zip(X, Y)] ))

##
## Generación de una malla de puntos
## y valor del SSE en cada punto
##
W0 = np.arange(-0.5, 3.0, 0.05)
W1 = np.arange(-0.5, 3.0, 0.05)
W0, W1 = np.meshgrid(W0, W1)
F = SSE(W0, W1)

##
##  Superficie de error
##
#plt.figure(2)
#fig = plt.figure(figsize=(7, 7))
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(W0, W1, F, cmap=cm.coolwarm, linewidth=1, antialiased=False)

##
## Contorno
##
def plot_contour():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal', 'box')
    ax.contour(W0, W1, F, levels=[0, 1, 2, 3, 5, 10, 20, 40, 60, 90])
    ax.grid()

plot_contour()

# Computar gradiente
def gSSE(w0, w1):
    ## calcula el vector de errores
    e = [(v - w0 - w1*u)  for u, v in zip(X, Y)]

    ## gradientes
    gw0 = -2 * sum(e)
    gw1 = -2 * sum([q*v for q, v in zip(e,X)])

    return (gw0, gw1)

def mejora(w0, w1, mu):
    ## computa el gradiente para los parámetros actuales
    gw0, gw1 = gSSE(w0, w1)

    ## realiza la corrección de los parámetros
    w0 = w0 - mu * gw0
    w1 = w1 - mu * gw1

    ## retorna los parámetros corregidos
    return (w0, w1)


## Punto de inicio
w0 = 0.5
w1 = 3.0

history_w0 = [w0]
history_w1 = [w1]
history_f  = [SSE(w0, w1)]

for epoch in range(20):
    w0, w1 = mejora(w0, w1, 0.005)
    history_w0.append(w0)
    history_w1.append(w1)
    history_f.append(SSE(w0, w1))

print('\nValores encontrados\n\n  w0 = {:f}\n  w1 = {:f}'.format(w0, w1))

plot_contour()
plt.plot(history_w0, history_w1, color='red');

##
##  A continuación se grafican la recta encontrada.
##

##  Se generan los puntos
z = np.linspace(0.0, 1.0)
y = w0 + w1 * z

## se grafican los datos originales
plt.figure(3)
plt.plot(X, Y, 'o');
## se grafica la recta encontrada
plt.plot(z, y, '-');

plt.show();


#Construir un modelo de regresión lineal que sirva como base para construir un modelo de redes neuronales artificiales.

m = linear_model.LinearRegression()
#print("# validation", cross_val_score(m, X, Y, cv=20).mean())
m.fit(np.array(X).reshape(-1, 1), Y)
y_pred = m.predict(np.array(X).reshape(-1, 1))
#print(y_pred)
print("Cross validation score: ", cross_val_score(m, np.array(X).reshape(-1, 1), Y, cv=20).mean())
plt.figure(4)
plt.plot(X, Y, '.r')
plt.plot(X, y_pred, '-b')

plt.show()

