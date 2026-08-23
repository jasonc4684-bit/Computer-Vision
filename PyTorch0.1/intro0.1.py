import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# using torch to manipulate/ find info about the data

#just 1D from 1 to 20
original = torch.arange(1., 21.)
print(original)

#reshape the tensor to 4x5
reshape = original.reshape(4, 5)

#similar to reshape in rearranging, but both share same memory
view = original.view(4,5)
print(reshape, reshape.shape)

#changes both view and original data
view[:, 0] = 30

print(view, view.shape)
print(original)

#stack 3 original data to form 1x3, dim=1 allows the data on the same row to be identical
stack = torch.stack([original, original, original], dim=0)
print(stack)