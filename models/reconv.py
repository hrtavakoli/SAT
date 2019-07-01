'''
Implementation of various recurrent convolution architectures

@author: Hamed R. Tavakoli
'''


import torch
import torch.nn as nn

import numpy as np


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, spatial_size):
        """

        :param input_channels: number of channels in the input tensor
        :param hidden_channels: number of expected output channels to the hidden space
        :param kernel_size: size of the convolution kernel
        :param spatial_size:
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        pad_size = int(np.floor(kernel_size/2))
        # define input gate weights
        self.W_xi = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True)
        self.W_hi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=False)

        self.W_ci = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define forget gate weights
        self.W_xf = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True)
        self.W_hf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=False)

        self.W_cf = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define output gate weights
        self.W_xo = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True)
        self.W_ho = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=False)

        self.W_co = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define cell weights
        self.W_xc = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=True)
        self.W_hc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=pad_size, bias=False)

    def _init_weights_(self):

        return 1

    def forward(self, x, h, c):
        ig = self.W_xi(x) + self.W_hi(h) + torch.mul(self.W_ci, c)
        ig = torch.sigmoid(ig)
        fg = self.W_xf(x) + self.W_hf(h) + torch.mul(self.W_cf, c)
        fg = torch.sigmoid(fg)
        c_new = torch.mul(fg, c) + torch.mul(ig, torch.tanh(self.W_xc(x) + self.W_hc(h)))
        output = torch.sigmoid(self.W_xo(x) + self.W_ho(h) + torch.mul(self.W_co, c))
        h_new = torch.mul(output, c_new)

        return [h_new, c_new]


class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=False):

        super(ConvLSTMCell, self).__init__()
        batch_size = input_size[0]




if __name__ == "__main__":

    input_x = torch.zeros([10, 100, 20, 30])
    input_c = torch.zeros([10, 50, 20, 30])
    input_h = torch.zeros([10, 50, 20, 30])
    model = ConvLSTMCell(100, 50, 3, [20, 30])
    model(input_x, input_h, input_c)