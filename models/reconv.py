'''
Implementation of various recurrent convolution architectures

@author: Hamed R. Tavakoli
'''


import torch
import torch.nn as nn

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

        # define input gate weights
        self.W_xi = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.W_hi = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)

        self.W_ci = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define forget gate weights
        self.W_xf = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.W_hf = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)

        self.W_cf = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define output gate weights
        self.W_xo = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.W_ho = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)

        self.W_co = nn.Parameter(torch.zeros(hidden_channels, spatial_size[0], spatial_size[1]))

        # define cell weights
        self.W_xc = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.W_hc = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=0, bias=False)


    def _init_weights_(self):

        @pass


    def forward(self, x, h, c):
        ig = nn.Sigmoid(
                self.W_xc(x) + self.W_hi(x) +  torch.mul(self.W_ci.expand_as(c), c)
        )