import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Adjusting tensor's dtype, 
#dtype usuall caused 3 errors: incorrect datatype, shapes, and device

output = torch.tensor([1,3,5,7], 
                    dtype=torch.float16,      # tensor type
                    device = None,            #
                    requires_grad=False)

print(output)



