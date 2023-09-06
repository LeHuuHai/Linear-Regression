import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/test.csv")
X = data.values[:, 0]
X = np.reshape(X, (-1, 1))
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
y = data.values[:, 1]
y = np.reshape(y, (-1,1))

def predict(Xbar, weight):
    return Xbar.dot(weight)

def grad(Xbar, y, weight):
    n = Xbar.shape[0]
    return 1/n * Xbar.T.dot(Xbar.dot(weight) - y)
def loss_function(Xbar, y, weight):
    n = Xbar.shape[0]
    return 0.5/n * np.linalg.norm(Xbar.dot(weight) - y)

def update_weight(Xbar, y, weight, eta):
    weight -= grad(Xbar, y, weight)*eta
    return weight

def train(Xbar, y, weight, eta, iter):
    loss = []
    for it in range(iter):
        loss_value = loss_function(Xbar, y, weight)
        loss.append(loss_value)
        weight = update_weight(Xbar, y, weight, eta)
        if np.linalg.norm(grad(Xbar, y, weight)) < 1e-3:
            break
    return weight, it, loss

weight = np.array([[0.001],[0.5]])
weight, iter, loss = train(Xbar, y, weight,0.0001, 100)

print(loss)
print(weight)
print(iter)

y_predict = predict(Xbar, weight)
plt.plot(X, y_predict, 'r')
plt.scatter(X, y)
plt.show()
