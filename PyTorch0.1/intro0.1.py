import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Adjusting tensor's dtype, 
#dtype usuall caused 3 errors: incorrect datatype, shapes, and device

#checks device 

output = torch.tensor([1,3,5,7], 
                    dtype=torch.float16,      # tensor type
                    device = None,            # which device to use, ex. GPU
                    requires_grad=False)      # checks gradients

output2 = torch.tensor([2,4,6,8], 
                    dtype=torch.int32,      # tensor type
                    device = None,            # which device to use, ex. GPU
                    requires_grad=False) 

print(output*output2)



