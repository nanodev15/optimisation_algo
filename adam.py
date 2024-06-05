def minibatchsgd_adam(X, y, lr, epochs, batch_size, beta1, beta2, epsilon):
    m, b = 0.5, 0.5 
    log, mse = [], []
    m_m = 0
    m_b = 0
    v_m = 0
    v_b = 0
    number_of_iterations = 0
    total_len = len(X)
    for epoch in range(epochs):
        
        for i in range(0, total_len, batch_size):
            number_of_iterations += 1
            Xs = X[i:i+batch_size]
            ys = y[i:i+batch_size]            
            N = len(Xs)
            f = ys - (m*Xs + b)
            gradient_m = (-2 * Xs.dot(f).sum() / N)
            gradient_b = (-2 * f.sum() / N)
                
            m_m = (beta1*m_m) + (1-beta1)*gradient_m 
            m_b = (beta1*m_b) + (1-beta1)*gradient_b
            v_m = (beta2*v_m) + (1 - beta2) * (gradient_m **2)
            v_b =  (beta2*v_b) + (1 - beta2) * (gradient_b **2)
            
            m_m_hat = m_m / (1- (beta1**number_of_iterations))
            m_b_hat = m_b / (1- (beta1**number_of_iterations)) 
            v_m_hat = v_m / (1-(beta2**number_of_iterations))
            v_b_hat = v_b / (1-(beta2**number_of_iterations))

            
            m = m - ( (lr * m_m_hat)/ ( (v_m_hat** 0.5) + epsilon))
            b = b - ( (lr * m_b_hat)/( (v_b_hat** 0.5) + epsilon))
            
            log.append((m, b))
            mse.append(mean_squared_error(y, (m*X + b)))     
            
    return number_of_iterations, m, b, log, mse
beta1 = 0.9
beta2 = 0.999
lr = 0.05 
epochs = 10
batch_size = 10
epsilon = 1e-08
no, m, b, log, mse = minibatchsgd_adam(X, y, lr, epochs, batch_size, beta1, beta2, epsilon)

total_updates = int(epochs*(len(X)/batch_size))
epochs = range(total_updates)
plt.figure(figsize=(5,5))
plt.scatter(epochs, mse, color = 'green')
plt.title("RQM vs Mises a jour - Adam")
plt.xlabel("Mises a jour")
plt.ylabel("RQM")
plt.show()