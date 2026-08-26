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
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))

        self.bias = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
    
