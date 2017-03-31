# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:29:21 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import excommon as com

dataset = com.load('ex2data1.txt')
X = np.mat(dataset)[:, 0:2]
y = np.mat(dataset)[:, 2:3]
plt.scatter(X[np.where(y==1)[0],0], X[np.where(y==1)[0],1], color='r', marker='+')
plt.scatter(X[np.where(y==0)[0],0], X[np.where(y==0)[0],1], color='g', marker='o')
plt.show()