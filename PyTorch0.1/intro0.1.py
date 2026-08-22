import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Adjusting tensor's dtype, 
#dtype usuall caused 3 errors: incorrect datatype, shapes, and device

#checks device 
# 2x3
output = torch.tensor([[1,3,5],
                        [2,4,6]], 
                    dtype=torch.int16,      # tensor type
                    device = None,            # which device to use, ex. GPU
                    requires_grad=False)      # checks gradients

#3x2
output2 = torch.tensor([[1,3],
                        [2,4],
                        [3,5]], 
                    dtype=torch.int16,      # tensor type
                    device = None,            # which device to use, ex. GPU
                    requires_grad=False)      # checks gradients

output3 = output2.T
#matrix multiplication, 
# inner dimensions must be identical and 
# shape will be the outer dimensions

#print(torch.mm(output, output2), output.shape, output2.shape) 
print(output2.shape, output3.shape)

