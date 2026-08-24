import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# reproducibility, also reduced randomness
random_seed = 42
torch.manual_seed(random_seed)


tensor_1 = torch.rand(3,4)
tensor_2 = torch.rand(3,4)

print(f"first tensor {tensor_1}, second tensor {tensor_2}")
print(tensor_1 == tensor_2)