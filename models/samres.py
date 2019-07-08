'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from models.reconv import ConvLSTM


class GaussPrior(nn.Module):
    """
        define a Gaussian prior
    """

    def __inint__(self, n_gp):
        """

        :param n_gp: number of Gaussian Priors
        :return:
        """

        self.ngp = n_gp

        self.W = nn.Parameter(torch.zeros(n_gp, 4))

        self._init_weights()

    def _init_weights(self):

        for i in range(self.ngp):
            torch.nn.init.uniform_(self.W[i, 0:1], 0.3, 0.7)
            torch.nn.init.uniform_(self.W[i, 2:3], 0.05, 0.3)

    def forward(self, x):

        [bs, _, h, w] = x.shape
        


class AttnConvLSTM(nn.Module):
    """
        Attentive Conv LSTM for SAMRES Model
    """
    def __init__(self, input_size, kernel_size, nstep=1):
        """
        :param input_size: the size of input in the form of [batch size x number of input channels x height x width]
        :param kernel_size:
        :param nstep: number of steps to compute the attentive mechanism
        """

        attn_size = input_size[1]
        self.bsize = input_size[0]
        self.Hsize = input_size[2]
        self.Wsize = input_size[3]

        self.convLSTM = ConvLSTM(input_size, attn_size, kernel_size)

        self.Wa = nn.Conv2d(input_size[1], input_size[1], kernel_size, padding=int(kernel_size/2), bias=True)
        self.Ua = nn.Conv2d(attn_size, attn_size, kernel_size, padding=int(kernel_size/2), bias=False)
        self.Va = nn.Conv2d(attn_size, 1, kernel_size, padding=int(kernel_size/2), bias=False)

        self.nstep = nstep

    def forward(self, x):

        output, [h, c] = self.convLSTM(x)

        for i in range(self.nstep):

            Zt = self.Va(F.tanh(self.Wa(x) + self.Ua(h)))
            At = F.softmax(Zt.view(self.bsize, -1), dim=1).view(self.bSize, 1, self.Hsize, self.Bsize)
            Xt = torch.mul(x, At)
            output, [h, c] = self.convLSTM(Xt)

        return output


class Model(nn.Module):
    # SAM ResNet model

    def __init__(self):
        super(Model, self).__init__()

        self.features = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
        self.dreduc = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder0 = nn.Sequential(
            nn.Conv2d(528, 512, kernel_size=5, stride=1, padding=2, dilation=4),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(528, 512, kernel_size=5, stride=1, padding=2, dilation=4),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        # we create a prior map of 6x8; the original models uses a 3x4 map. We We think this is more effective
        self.prior = nn.Parameter(torch.ones((1, 1, 6, 8), requires_grad=True))

    def forward(self, inputs):

        inputs = self.features(inputs)
        inputs = self.dreduc(inputs)

        output = F.dropout(inputs, p=0.5)
        output = self.decoder(output)
        output = output * F.interpolate(self.prior, size=(output.shape[2], output.shape[3]))
        return output


if __name__ == "__main__":
    data = torch.zeros(1, 3, 240, 320)
    m = Model()
    a = m(data)
    print(a.shape)