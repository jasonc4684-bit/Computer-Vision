import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from torch import nn # contains building block for pytorch's neural network

# building model 
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# data split
training_set = int(0.8 * len(X))
training_X, training_y = X[:training_set], y[:training_set] 

test_X, test_y = X[training_set:], y[training_set:] 

class LinearRegModel(nn.Module): # inherit nn.Module for useful prebuilt tools
    def __init__(self):
        super().__init__() # access all methods from nn.Module

        #Initializing 

        #nn.Paramter auto assign and train the left_side to deep learning afterward
        self.weights = nn.Parameter(torch.randn(1,      # starts off a random value
                                                requires_grad=True, #optimizes the loss function
                                                dtype=torch.float))  #default to float32

        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
    
    #Required func for pytorch deeplearning, to override
    def forward(self, x:torch.Tensor) -> torch.Tensor: #expects input x to be same with output, torch.Tensor
        return self.weights * x + self.bias         # linear regression 

        
# torch.optim() optimizes/help with gradient descent
#torch.utils.data.Dataset() map the key and sample pair 
#torch.utils.data.Dataloader() iterate over the torch Dataset

# setting seed
torch.manual_seed(42)
model_0 = LinearRegModel()
print(list(model_0.parameters()), model_0.state_dict()) 


def plotdata(train_data=training_X, 
            train_label=training_y,
            test_data=test_X,
            test_label=test_y,
            prediction=None):
    plt.figure(figsize=(10,7))

    plt.scatter(training_X, training_y, s=4, c='b', label='Training data')

    plt.scatter(test_X, test_y, s=4, c='r', label='Testing data')

    if prediction is not None:
        plt.scatter(test_X, prediction, s=4, c='g', label="Prediction")

    plt.legend(prop={"size":14})
    plt.show()

with torch.inference_mode(): # disabiling the gradeint tracking if not training
    y_predic = model_0(test_X)

print(y_predic)  

plotdata(prediction=y_predic) # plotting the differences between known data and pretrained data

# loss/cost/criteria function measures the distance apart is the prediction
# optimizers - account the loss and adjust the model's parameters

