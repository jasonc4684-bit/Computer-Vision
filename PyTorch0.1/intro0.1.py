import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# using torch to manipulate/ find info about the data

x = torch.tensor([[5,6,7]])
print(x.shape)

print(x.squeeze().shape)