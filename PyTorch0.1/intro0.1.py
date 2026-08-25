import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from torch import nn # contains building block for pytorch's neural network

# Preparing and loading

# turns all data into numerical representation to the model (layers)
# learns from the data's pattern and present output

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# data split
training_set = (0.8 * len(X))
traing_X, training_y = X[:testing_set], y[:testing_set] 

test_X, test_y = X[testing_set:], y[testing_set:] 

def plotdata(train_data=traing_X, 
            train_label=training_y,
            test_data=test_x,
            test_label=test_y,
            prediction=None):
    plt.figure(figsize=(10,7))

    plt_train = plt.scatter(traing_X, training_y, s=4, c='b', label='Plotting training data')

    plt_test = plt.scatter(test_X, test_y, s=4, c='r', label='Plotting testing data')