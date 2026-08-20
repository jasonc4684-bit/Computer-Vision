import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Creating random tensors
imgHeight = 224
imgWidth = 224
Matrix = torch.rand(size=(3,imgHeight, imgWidth))
print(Matrix.ndim, Matrix.shape)
