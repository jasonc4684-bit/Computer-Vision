import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Indexing in PyTorch

x = torch.arange(1,10).reshape(1,3,3)

print(x[0,2,2])
print(x[:, :, 2])