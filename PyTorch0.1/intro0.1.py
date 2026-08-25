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

print(X[:10], y[:10], len(X), len(y))