import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Accessing/ Running on GPU/ required to transport code to google colab

# check gpu availability

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if larger gpu is needed, use torch.cuda.device_count() to check for additional gpu

tensor = torch.tensor([1,2,3])
tensor.to(device) # prefer if cuda is available

#print(tensor, tensor.device)

# to fix gpu error with numpy, use .cpu() to switch to cpu

# extracurriculum 
# random tensor with shape (7,7)
tensor_1 = torch.rand(7,7)
print(tensor_1.shape)

# matrix multiplication from 2 with another with shape (1,7)
print()