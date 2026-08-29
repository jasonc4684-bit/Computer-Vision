import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

from torch import nn  # contains building block for pytorch's neural network

"""
training loop steps
0. loop over the data
1. forward pass, utilizing `forward` function
2. loss calculation
3. optimizer zero grad
4. loss backward, backpropagation
5. optimizer step, adjust parameters and improve - loss gradient descent
"""

# building model
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# data split
training_set = int(0.8 * len(X))
training_X, training_y = X[:training_set], y[:training_set]

test_X, test_y = X[training_set:], y[training_set:]


class LinearRegModel(nn.Module):  # inherit nn.Module for useful prebuilt tools
    def __init__(self):
        super().__init__()  # access all methods from nn.Module

        # Initializing
        # nn.Paramter auto assign and train the left_side to deep learning afterward
        self.weights = nn.Parameter(
            torch.randn(
                1,  # starts off a random value
                requires_grad=True,  # optimizes the loss function
                dtype=torch.float,
            )
        )  # default to float32

        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    # Required func for pytorch deeplearning, to override
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # expects input x to be same with output, torch.Tensor
        return self.weights * x + self.bias  # linear regression

'''
torch.optim() optimizes/help with gradient descent
torch.utils.data.Dataset() map the key and sample pair
 torch.utils.data.Dataloader() iterate over the torch Dataset
'''

# setting seed
torch.manual_seed(42)
model_0 = LinearRegModel()
print(list(model_0.parameters()), model_0.state_dict())

def plotdata(
    train_data=training_X,
    train_label=training_y,
    test_data=test_X,
    test_label=test_y,
    prediction=None,
):
    plt.figure(figsize=(10, 7))

    plt.scatter(training_X, training_y, s=4, c="b", label="Training data")

    plt.scatter(test_X, test_y, s=4, c="r", label="Testing data")

    if prediction is not None:
        plt.scatter(test_X, prediction, s=4, c="g", label="Prediction")

    plt.legend(prop={"size": 14})
    plt.savefig('my_plot.png')


with torch.inference_mode():  # disabiling the gradeint tracking if not training
    y_predic = model_0(test_X)

print(y_predic)

plotdata(
    prediction=y_predic
)  # plotting the differences between known data and pretrained data

# loss/cost/criteria function measures the distance apart is the prediction
# ex. nn.L1Loss() uses MAE(mean abs. error) between x and y, which nn.MSELoss uses mean^2

# nn.L1Loss() = torch.mean(abs(y_predic - y_test))
loss_fn = nn.L1Loss()

# optimizers - account the loss and adjust the model's parameters
optim_fn = torch.optim.SGD( # stochastic gradient descent optimizer
    params=model_0.parameters(), lr=0.01   # learning rate, propotion to change in parameters
) 

# epochs, loops
epochs = 200

epoch_list = []
loss_list = []
test_loss_list = []

for epoch in range(epochs):
    model_0.train()  # auto enables required-gradients
    # 1. forward pass
    y_pred = model_0(training_X)

    # 2. loss func
    loss = loss_fn(y_pred, training_y)  # (input, target)

    # 3. zero grad
    optim_fn.zero_grad()

    # 4. backpropagation
    loss.backward()

    # 5. step
    optim_fn.step()

    model_0.eval()  # disables required-gradients
    with torch.inference_mode():
        # forward pass 
        y_pred = model_0(test_X)
        # loss
        loss_test = loss_fn(y_pred, test_y.type(torch.float))

    # print current loss
    if epoch % 10 == 0:
        epoch_list.append(epoch)
        loss_list.append(loss.detach().numpy())
        test_loss_list.append(loss_test.detach().numpy())

        print(f"epoch {epoch}, loss {loss}, test loss {loss_test}")

        print(model_0.state_dict()) #print the parameters' values

plotdata(prediction=y_pred) 

plt.figure(figsize=(10, 7))
plt.plot(epoch_list, loss_list, label="loss from epoch")
plt.plot(epoch_list, test_loss_list, label="test_loss from epoch")
plt.title("loss values from each generated epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend() 
plt.savefig('my_train_plot.png')

'''
saving/loading training data
0. torch.save() - return python's pickle format
1. torch.load() - load a saved object
2. torch.nn.Module.load_state_dict() - load the model's saved state dictionary
'''
#creating path
model_path = Path("model_0")
model_path.mkdir(parents=True, exist_ok=True)

#model save path
model_name = "01_model_linear_regression_for_73_split.pt"

model_save_path = model_path / model_name

#saving current model's state dict
print(f"Saving current state dictionary to {model_save_path} ....")
torch.save(obj=model_0.state_dict(), f=model_save_path)