import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# using torch to manipulate/ find info about the data

output = torch.arange(1., 20.)
print(output)
output2 = output.reshape(19, 1)
print(output2, output2.shape)