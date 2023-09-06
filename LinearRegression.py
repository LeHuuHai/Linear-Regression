import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataFrame = pd.read_csv('data/dataL.csv')
X = dataFrame.values[:, 1]
y = dataFrame.values[:, 3]


def predict(new_radio, weight, bias):
    return weight*new_radio + bias

def cost_function(X, y, weight, bias):
    n = len(X)
    sum_loss = 0
    for i in range(n):
        sum_loss += (y[i] - (weight*X[i]+bias))**2
    return sum_loss/n

def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_tmp = 0.0
    bias_tmp = 0.0
    for i in range(n):
        weight_tmp += -2*X[i]*(y[i] - (weight*X[i]+bias))
        bias_tmp +=-2*(y[i] - (weight*X[i]+bias))
    weight -= (weight_tmp/n)*learning_rate
    bias -= (bias/n)*learning_rate
    return weight, bias

def train(X, y, weight, bias, learning_rate, iter):
    cost_his = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        cost = cost_function(X, y, weight, bias)
        cost_his.append(cost)
    return weight, bias, cost_his

weight, bias, cost = train(X, y, 0.03, 0.0014, 0.001, 60)
print("weight: ", weight)
print("bias: ", bias)
print("cost: ", cost)
print("predict: ")
print(predict(19, weight, bias))

plt.scatter(X, y, marker='o')
x_line = np.linspace(0,60,100)
def f(x, weight, bias):
    return weight*x+bias
y_line = f(x_line, weight, bias)
plt.plot(x_line, y_line)
plt.show()
