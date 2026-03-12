import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1,1)

    m, n = X.shape

    w = np.zeros((n,1),float)
    b = 0.0
    
    for i in range(steps):
        z = np.dot(X, w) + b
        p = sigmoid(z)

        g_w = (1/m) * np.dot(X.T, (p - y))
        g_b = np.mean(p - y)
        
        w -= lr * g_w
        b -= lr * g_b

    return w.reshape(n), b