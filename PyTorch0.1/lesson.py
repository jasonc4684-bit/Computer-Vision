import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from torch import nn # contains building block for pytorch's neural network

# building model 

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

        