import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# using torch to manipulate/ find info about the data

original = torch.arange(1., 21.)
print(original)

reshape = original.reshape(4, 5)
view = original.view(4,5)

print(reshape, reshape.shape)

view[:, 0] = 30

print(view, view.shape)
print(original)