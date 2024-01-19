import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

##==============PyTorch Workflow==============

#--lineral regresion--   Y=a+bX

#create *known* parameters

weight = 0.7
bias = 0.3

stat =0
end =1
step =0.02
X = torch.arange(stat,end,step).unsqueeze(dim=1)
y = weight *X + bias

# print(f"x: {X[:10]},\n y: {y[:10]},\n len X: {len(X)},\n len y: {len(y)}")

## Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]  # : behind means onwards

# Function to plot predictions
def plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=None):
    """Plots training data, test data, and predictions."""
    plt.figure(figsize=(10, 7))

    # Train data
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # Test data
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    # Are there predictions?
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

# plot_predictions()

#Create linear regresion model

class LinearRegressionModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True, dtype=torch.float))
        self.bias =nn.Parameter(torch.randn(1, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()

print(model_0.state_dict())
