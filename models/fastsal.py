'''
    This is an implementation based on  the Light Weight Saleincy of Shanghua with some modifications

    @author: Xiao Shanghua

    File modified significantly to fit to the framework
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

INPLACE = True


class InvertedResidual(nn.Module):

    def __init__(self, c_in, c_out, exp):
        super(InvertedResidual, self).__init__()
        self.use_res = c_in == c_out
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, exp, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, exp, kernel_size=3, stride=1, padding=1, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            nn.ReLU6(inplace=INPLACE),
            nn.Conv2d(exp, c_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fine_net = self.SubNet()
        self.coarse_net = self.SubNet()

    def forward(self, fine_batch):

        fine = self.fine_net(fine_batch)
        coarse = F.interpolate(self.coarse_net(F.interpolate(fine_batch, scale_factor=0.5)), scale_factor=2)
        mul = coarse * fine
        return mul

    # def SubNet(self):
    #     return nn.Sequential(
    #         self.Conv(3, 32, 3, 1),
    #         self.Pool('max'),
    #         self.ConvDWS(32, 16, 64),
    #         self.ConvDWS(16, 16, 64),
    #         self.Pool('max'),
    #         self.ConvDWS(16, 24, 96),
    #         self.ConvDWS(24, 24, 96),
    #         self.ConvDWS(24, 24, 96),
    #         self.Pool('max'),
    #         self.ConvDWS(24, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 32, 128),
    #         self.ConvDWS(32, 64, 256),
    #         self.ConvDWS(64, 64, 256),
    #         self.ConvDWS(64, 128, 512),
    #         self.Conv(128, 1, 1, 0)
    #     )

    def SubNet(self):
        return nn.Sequential(
            self.Conv(3, 32, 3, 1),
            self.Pool('max'),
            InvertedResidual(32, 16, 64),
            InvertedResidual(16, 16, 64),
            self.Pool('max'),
            InvertedResidual(16, 24, 96),
            InvertedResidual(24, 24, 96),
            InvertedResidual(24, 24, 96),
            self.Pool('max'),
            InvertedResidual(24, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 32, 128),
            InvertedResidual(32, 64, 256),
            InvertedResidual(64, 64, 256),
            InvertedResidual(64, 128, 512),
            self.Conv(128, 1, 1, 0)
        )

    # def OverfitNet(self):
    #
    #     return nn.Sequential(
    #         self.Conv(3, 32, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(32, 16, 3, 1),
    #         self.Conv(16, 16, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(16, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(24, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 64, 3, 1),
    #         self.Conv(64, 64, 3, 1),
    #         self.Conv(64, 128, 3, 1),
    #         self.Conv(128, 1, 1, 0)
    #     )

    # def TinyNet(self):
    #     return nn.Sequential(
    #         self.Conv(3, 32, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(32, 16, 3, 1),
    #         self.Conv(16, 16, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(16, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(24, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 64, 3, 1),
    #         self.Conv(64, 64, 3, 1),
    #         self.Conv(64, 128, 3, 1),
    #         self.Conv(128, 1, 1, 0)
    #     )
    #
    # def MicroNet(self):
    #     return nn.Sequential(
    #         self.Conv(3, 32, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(32, 16, 3, 1),
    #         self.Conv(16, 16, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(16, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Conv(24, 24, 3, 1),
    #         self.Pool('max'),
    #         self.Conv(24, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 32, 3, 1),
    #         self.Conv(32, 1, 1, 0)
    #     )

    # def ConvDWS(self, c_in, c_out, exp):
    #     return nn.Sequential(
    #         nn.Conv2d(c_in, exp, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
    #         nn.BatchNorm2d(exp),
    #         nn.ReLU6(inplace=INPLACE),
    #         nn.Conv2d(exp, exp, kernel_size=3, stride=1, padding=1, groups=exp, bias=False),
    #         nn.BatchNorm2d(exp),
    #         nn.ReLU6(inplace=INPLACE),
    #         nn.Conv2d(exp, c_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
    #         nn.BatchNorm2d(c_out),
    #         # nn.ReLU6(inplace=INPLACE)
    #     )
    #
    def Conv(self, c_in, c_out, ksize, pad):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ksize, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU6(inplace=INPLACE)
        )

    def Pool(self, typ):
        if typ == 'max':
            return nn.MaxPool2d(2, stride=2)


if __name__ == "__main__":
    sample_input = torch.zeros(1, 3, 512, 640).cuda()
    model = Model().cuda()
    a = model(sample_input)
    print(a.shape)
    model.train()
    n_param = sum(p.numel() for p in model.parameters())
    print(model(sample_input).shape)
    print(n_param)