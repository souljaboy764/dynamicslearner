import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden=100):
        super(Model, self).__init__()
        self.__hidden = nn.Linear(num_inputs, num_hidden)
        self.__output = nn.Linear(num_hidden, num_outputs)
        self.__activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.__output(self.__activation(self.__hidden(x)))