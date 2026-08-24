import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# PyTorch & NumPy

# utilize torch.from_numpy() in translating NumPy data to PyTorch for deep learning , different memory
# use torch.Tensor.numpy to translating data to NumPy

np_data = np.arange(1, 10)
print(np_data)

to_torch = torch.from_numpy(np_data) # default dtype is int64 from numpy
print(to_torch)

torch_data = torch.arange(1,10)
print(torch_data)

to_np = torch.Tensor.numpy(to_torch)
print(to_np)