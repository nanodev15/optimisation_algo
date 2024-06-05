import numpy as np
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

def SGD(X, y, learning_rate, epochs): 
    m, b = 0.5, 0.5 # initialisation des paramètres
    # log enregistrre les valeurs m et b pour differents paramètres
    # mse enregistre l'erreur
    log, mse = [], [] 
    
    for _ in range(epochs):       
	indexes = np.random.randint(0, len(X)) # echantillons simples        
        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
	N = len(X)
	f = ys - (m*Xs + b)

	# Updating parameters m and b
        m -= lr * (-2 * Xs*(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)

        log.append((m, b))
        mse.append(mean_squared_error(y, m*X+b))        

     return m, b, log, mse



# set seed for reproducibilty
np.random.seed(1) 
num_samples = 100
X = np.random.uniform(-1.,1.,num_samples)
m = 5
b = 1
y = m*X +b # y = 2.5X + 1

X

  
epochs = 500
lr = 0.85
m, b, log, mse = SGD(X, y, lr, epochs)
m, b, len(mse)

epochs = range(epochs) 
plt.figure(figsize=(5,5))
plt.title("EQM contre epochs - SGD")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.scatter(epochs, mse, color = 'green')
plt.show()