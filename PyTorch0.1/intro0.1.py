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
#tensor.to(device) if cuda is available

print(tensor, tensor.device)