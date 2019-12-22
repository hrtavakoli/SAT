'''
Implementation of MLNet model by Cornia et al.

@author: Hamed R. Tavakoli
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16


class Model(nn.Module):
    # mlnet model

    def __init__(self):
        super(Model, self).__init__()

        self.features_0 = nn.Sequential(*nn.ModuleList(list(vgg16(pretrained=True).features)[:2]))
        self.features_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(64, 786, kernel_size=1, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(786, 786, kernel_size=3, stride=1, groups=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.3)

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(786, 1, kernel_size=1, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.initialize()
        for param in self.features_0.parameters():
            param.requires_grad = False

    def initialize(self):
        nn.init.kaiming_normal_(self.features_1[1].weight)
        nn.init.kaiming_normal_(self.features_1[4].weight)
        nn.init.kaiming_normal_(self.decoder[0].weight)

    def forward(self, inputs):

        output = self.features_0(inputs)
        output = self.features_1(output)
        output = self.decoder(output)
        output = F.dropout(output, p=0.3)
        return output


if __name__ == "__main__":
    data = torch.ones(1, 3, 320, 480).cuda()
    model = Model().cuda()
    output = model(data)
    #print(output)
    model.train()
    n_param = sum(p.numel() for p in model.parameters())
    print(model(data).shape)
    print(n_param)