'''
Implementation of various recurrent convolution architectures

@author: Hamed R. Tavakoli
'''


import torch
import torch.nn as nn

class AttnConvLSTM(nn.Module):
    """
        Attentive Conv LSTM
    """

    def __init__(self, input_channels, out_channels, kernel_size, hidden_size, return_sequences=False, bias=True):
        """

        :param input_channels: number of channels in the input tensor
        :param out_channels: number of expected output channels
        :param kernel_size: size of the convolution kernel
        :param hidden_size:
        :param bias:
        """
        super(AttnConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.conv_cell = nn.conv2d(input_channels=self.input_channels, out_channels=self.out_channels)

