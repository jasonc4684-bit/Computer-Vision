import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Accessing/ Running on GPU/ required to transport code to google colab

# check gpu availability

device = 'cuda' if torch.cuda.is_available() else 'cpu'