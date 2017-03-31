# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:29:21 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt

def load(file_name):
    with open(file_name, 'r') as file:
        dataset = []
        for line in file:
            eles = line.split(',')
            if len(eles) < 1:
                continue
            eles = list(map(float, eles))
            dataset.append(eles)
    return np.array(dataset)
    
def compute_cost(X, y, theta):
    m = len(y)
    j = X*theta - y
    return (1/(2*m))*np.sum(np.power(j, 2))
    
def gradientdescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros([num_iters, 1])
    for i in range(0, num_iters):
        j = X * theta - y
        theta = theta - alpha * (1/m) * np.sum(np.multiply(j, X))
        j_history[i] = compute_cost(X, y, theta)    
    return theta, j_history
    
def feature_normalize(x):
    mu = np.mean(x, 1)
    sigma = np.std(x, 1)
    x = np.divide(x - mu, sigma)
    return x, mu, sigma

#plot data
dataset = com.load('ex1data1.txt')
# plt.scatter([x[0] for x in dataset], [y[1] for y in dataset], marker = 'x', color='red')
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# plt.show()

#Gradient descent
x = (np.mat(dataset))[:,0]
y = (np.mat(dataset))[:,1]
X = np.insert(x, 0, 1, 1)
theta = np.matrix(np.zeros([2,1]))
num_iters = 1500
alpha = 0.01
theta, j_history = gradientdescent(X, y, theta, alpha, num_iters)
# print('j_history:', j_history)
# print('theta', theta)

x_data = [x for x in range(0,25)]
y_data = (theta[0] + theta[1] * x_data).tolist()[0]
# plt.plot(x_data, y_data)
# plt.show()


#plot theta 
theta0_val = np.linspace(-10, 10, 100)
theta1_val = np.linspace(-1, 4, 100)
j_vals = np.zeros([len(theta0_val), len(theta1_val)])

for i, t0 in enumerate(theta0_val):
    for j, t1 in enumerate(theta1_val):
        current_theta = np.mat([[t0], [t1]])
        j_vals[i, j] = compute_cost(X, y, current_theta)        

X_theta0, Y_theta1 = np.meshgrid(theta0_val, theta1_val)
# plt.contour(theta0_val, theta1_val, j_vals, 200)
# plt.show()

#multi variable
muldataset = com.load('ex1data2.txt')
mul_x = np.mat(muldataset)[:, 0: 2]
mul_y = np.mat(muldataset)[:, 2: 3]
theta = np.matrix(np.zeros([3,1]))
alpha = 0.1;
num_iters = 400;

#Scale features and set them to zero mean
mul_X, mu, sigma = feature_normalize(mul_x)
mul_X = np.insert(mul_X, 0, 1, 1)

theta, j_history = gradientdescent(mul_X, mul_y, theta, alpha, num_iters)
# plt.plot([x for x in range(0, num_iters)], j_history)
# plt.show()
mul_y_diff = np.insert(np.divide(mul_x - mu, sigma), 0, 1, 1) * theta - mul_y
print(mul_y_diff)
