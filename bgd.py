import numpy as np
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

def batch_gradient_descent(X, y, learning_rate, epochs): 
    m, b = 0.5, 0.5 # initialisation des paramètres
    # log enregistrre les valeurs m et b pour differents paramètres
    # mse enregistre l'erreur
    log, mse = [], [] 
    N = len(X) # Nombre d'echantillons
    
    for _ in range(epochs):               
        f = y - (m*X + b)   
        # Mettre a jour m et b
        m -= learning_rate * (-2 * X.dot(f).sum() / N)
        b -= learning_rate * (-2 * f.sum() / N)
    
        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    return m, b, log, mse


# set seed for reproducibilty
np.random.seed(1) 
num_samples = 100
X = np.random.uniform(-1.,1.,num_samples)
m = 5
b = 1
y = m*X +b # y = 2.5X + 1

X

lr = 0.1
epochs = 75
m, b, log, mse = batch_gradient_descent(X, y, lr, epochs)

print(m)
print(b)
# print(log)
# print(mse)
print(len(mse))

epochs = range(epochs) 
plt.figure(figsize=(5,5))
plt.scatter(epochs, mse, color = 'green')
plt.title("MSE contre epochs - Batch Gradient Descent")
plt.xlabel("epochs")
plt.ylabel("EQM")
plt.show()