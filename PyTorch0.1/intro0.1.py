import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# using torch to manipulate/ find info about the data

output = torch.arange(10, 150, 23)

print(output.argmin(), torch.argmax(output), output[0], output[6])