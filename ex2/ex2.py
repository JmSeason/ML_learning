# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:29:21 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import excommon as com
import scipy.optimize as op

def sigmoid_func(x):
    return 1.0/(1.0 + np.exp(-x))

def gradient(theta, X, y):
    m = y.shape[1]
    h = sigmoid_func(theta * X)
    grad = (1/m)*np.sum(np.multiply(h - y, X), 1)
    return grad.flatten()

def cost_func(theta, X, y):
    m = y.shape[1]
    h = sigmoid_func(theta * X)
    j = (1/m)*np.sum(-np.multiply(y, np.log(h)) - np.multiply((1-y), np.log(1-h)))
    return j
    
def plot_decision_boundary(theta, X):
    plot_xy = np.linspace(np.min(X[1]), np.max(X), 100)
    plot_z = np.eye(len(plot_xy))
    for i, x in enumerate(plot_xy):
        for j, y in enumerate(plot_xy):
            current_x_matrix = np.mat([[1], [x], [y]])
            plot_z[i, j] = theta * current_x_matrix
    plt.contour(plot_xy, plot_xy, plot_z, [0])
    plt.show()
    
def map_feature(X):
    out = np.mat(np.ones(X.shape[1]))
    for i in range(1, 7):
        for j in range(0, i + 1):
            out = np.append(out,
                            np.multiply(np.power(X[0], i - j), np.power(X[1], j)),
                            0)
    return out
    
def cost_func_reg(theta, X, y, lambda_value):
    m = y.shape[1]
    h = sigmoid_func(theta * X)
    j = (1/m) * np.sum(-np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))) + lambda_value/(2*m) * np.sum(np.power(theta, 2))
    return j

def grad_reg(theta, X, y, lambda_value):
    m = y.shape[1]
    h = sigmoid_func(theta * X)
    grad = ((1/m) * np.sum(np.multiply(h-y, X), 1)).flatten() + lambda_value/m*theta
    grad[0] = grad[0] - lambda_value/m*theta[0]
    return grad
    
def plot_decision_boundary_reg(theta, X):
    plot_xy = np.linspace(np.min(X), np.max(X), 10)
    plot_z = np.eye(len(plot_xy))
    for i, x in enumerate(plot_xy):
        for j, y in enumerate(plot_xy):
            current_x_matrix = map_feature(np.mat([[x], [y]]))
            plot_z[i, j] = theta * current_x_matrix
    plt.contour(plot_xy, plot_xy, plot_z, [0])
    plt.show()
    

#Logistic Regression
dataset = com.load('ex2data1.txt')
X = np.mat(dataset)[:, 0:2]
y = np.mat(dataset)[:, 2:3]

#show data
# plt.scatter(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], color='r', marker='+')
# plt.scatter(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], color='g', marker='o')
# plt.show()

#calcurate the best theta
X = np.insert(X, 0, 1, axis=1).T
y = y.T
theta = np.mat(np.zeros(len(X)))
cost = cost_func(theta, X, y)
grad = gradient(theta, X, y)
print('cost before', cost)
print('grad before', grad)

theta = op.minimize(fun = cost_func,
                    x0 = theta,
                    args = (X, y),
                    method = 'TNC',
                    jac = gradient,
                    options = {'maxiter':400}).x;

print('theta after', theta)

#show halving line
#plot_decision_boundary(theta, X)

#Regularized logistic regression
dataset = com.load('ex2data2.txt')
X = np.mat(dataset)[:, 0:2]
y = np.mat(dataset)[:, 2:3]
#show data
plt.scatter(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], color='r', marker='+')
plt.scatter(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], color='g', marker='o')
# plt.show()
X = X.T
y = y.T
lambda_value = 5
X_mf = map_feature(X)
theta = np.mat(np.zeros(X_mf.shape[0]))
cost = cost_func_reg(theta, X_mf, y, lambda_value)
grad = grad_reg(theta, X_mf, y, lambda_value)
print('reg cost before', cost)
print('reg grad before', grad)
#search the best theta
theta = op.minimize(fun = cost_func_reg,
                    x0 = theta,
                    args = (X_mf, y, lambda_value),
                    method = 'TNC',
                    jac = grad_reg,
                    options = {'maxiter':400}).x;

print('reg theta after', theta)
#plot halving line
plot_decision_boundary_reg(theta, X)
