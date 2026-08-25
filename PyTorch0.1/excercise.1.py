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
print(f"tensor with shape (7,7) {tensor_1} shape : {tensor_1.shape}")

# matrix multiplication from 2 with another with shape (1,7)
tensor_2 = torch.rand(1,7)
matmul_1_and_2 = torch.mm(tensor_1, tensor_2.T)
print(f"matrix multiplication from 2 with another with shape (1,7): {matmul_1_and_2} with shape {matmul_1_and_2.shape}")

# using seed() to do excercise 1 and 2

random_seed = 0
torch.manual_seed(random_seed)
tensor_3 = torch.rand(7,7)
print(f"using seed() to do excercise 1 and 2 {tensor_3} with shape {tensor_3.shape}")

tensor_4 = torch.rand(1,7)
matmul_3_and_4 = torch.mm(tensor_3, tensor_4.T)
print(matmul_3_and_4, matmul_3_and_4.shape)

print(f"sum of matmul_3_and_4: {sum(matmul_3_and_4)}, max: {max(matmul_3_and_4)}") 
print(f"min of matmul_3_and_4: {min(matmul_3_and_4)}, mean: {matmul_3_and_4.type(torch.float32).mean()}")

print(f"max index of matmul_3_and_4: {matmul_3_and_4.argmax()}, min index: {matmul_3_and_4.argmin()}")

torch.manual_seed(7)

torch_5 = torch.rand(1,1,1,10)
torch_6 = torch_5.squeeze()
print(f"tensor with shape (1,1,1,10): {torch_5} shape : {torch_5.shape}, after squeezed, tensor: {torch_6} with shape {torch_6.shape}")


