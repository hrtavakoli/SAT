
# Implementation of MLNet model by Cornia et al.

import torch
import torch.nn as nn

from torchvision.models import vgg16


class Model(nn.Module):
    # mlnet model

    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.ModuleList(list(vgg16(pretrained=True).features)[:-1])


    def forward(self, inputs):
        pass

