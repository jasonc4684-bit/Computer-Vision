import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Indexing in PyTorch

x = torch.arange(1,31).reshape(3,2,5)

print(x[1:, :, 3])