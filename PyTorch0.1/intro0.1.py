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
tensor_2 = torch.rand(1,7)
matmul_1_and_2 = torch.mm(tensor_1, tensor_2.T)
print(matmul_1_and_2, matmul_1_and_2.shape)

# using seed() to do excercise 1 and 2

random_seed = 0
torch.manual_seed(random_seed)
tensor_1 = torch.rand(7,7)
print(tensor_1.shape)

tensor_2 = torch.rand(1,7)
matmul_1_and_2 = torch.mm(tensor_1, tensor_2.T)
print(matmul_1_and_2, matmul_1_and_2.shape)

print(sum(matmul_1_and_2), max(matmul_1_and_2), min(matmul_1_and_2), 
        matmul_1_and_2.type(torch.float32).mean())

print(matmul_1_and_2.argmax(), matmul_1_and_2.argmin())

torch.manual_seed(7)

torch_3 = torch.rand(1,1,1,10)
torch_4 = torch_3.squeeze()
print(torch_3, torch_3.shape, torch_4, torch_4.shape)


