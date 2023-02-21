import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=100):
        super(Model, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.hidden = nn.Sequential(nn.Linear(num_inputs, num_hidden), self.activation, nn.Linear(num_hidden, num_hidden))
        self.output = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))