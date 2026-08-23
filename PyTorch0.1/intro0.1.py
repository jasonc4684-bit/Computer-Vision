import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# squeeze() remove one dimesion

x = torch.tensor([[5,6,7]])
print(f"X: {x}, and its shape is {x.shape}")

squeezed_x = x.squeeze()
print(f"with added squeeze(), x changed to {squeezed_x} and shape is {squeezed_x.shape}")

y = torch.tensor([5,6,7])
unsqueeze_y = y.unsqueeze(dim=0)
print(f"y has tensor {y} with shape {y.shape}")
print(f"with unsquezzed, {unsqueeze_y} with shape of {unsqueeze_y.shape}")
