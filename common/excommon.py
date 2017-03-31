# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:29:21 2017

@author: Administrator
"""

import numpy as np

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