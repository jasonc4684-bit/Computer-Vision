import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# PyTorch & NumPy

# utilize torch.from_numpy() in translating NumPy data to PyTorch for deep learning 
# use torch.Tensor.numpy to translating data to NumPy

np_data = np.arange(1, 10)
print(np_data)

to_torch = torch.from_numpy(np_data)
print(to_torch)