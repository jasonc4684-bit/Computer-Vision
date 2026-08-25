import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from torch import nn # contains building block for pytorch's neural network

# Preparing and loading

# turns all data into numerical representation to the model (layers)
# learns from the data's pattern and present output