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

with torch.inference_mode():
    y_predic = model_0(te)