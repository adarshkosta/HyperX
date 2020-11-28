import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import time
# __all__ = ['net']


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()

    def forward(self, x):
        out = self.ConvBNReLU1(x)
        out = self.InvertedResidual1(out)
        out = self.InvertedResidual2(out)
        residual = out.clone()
        # print(residual.shape)
        # print((self.InvertedResidual3(out)).shape)
        out = self.InvertedResidual3(out) + residual
        out = self.InvertedResidual4(out)
        residual = out.clone()
        out = self.InvertedResidual5(out) + residual
        residual = out.clone()
        out = self.InvertedResidual6(out) + residual
        out = self.InvertedResidual7(out)
        residual = out.clone()
        out = self.InvertedResidual8(out) + residual
        residual = out.clone()
        out = self.InvertedResidual9(out) + residual
        residual = out.clone()
        out = self.InvertedResidual10(out) + residual
        out = self.InvertedResidual11(out)
        residual = out.clone()
        out = self.InvertedResidual12(out) + residual
        residual = out.clone()
        out = self.InvertedResidual13(out) + residual
        out = self.InvertedResidual14(out)
        residual = out.clone()
        out = self.InvertedResidual15(out) + residual
        residual = out.clone()
        out = self.InvertedResidual16(out) + residual
        out = self.InvertedResidual17(out)
        out = self.ConvBNReLU2(out)

        # out = self.avgpool(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        out = self.dropout1(out)
        out = self.fc(out)

        return out


class mobilenet_v2(mobilenet):
    def __init__(self, num_classes=1000):
        super(mobilenet_v2, self).__init__()

        # block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        self.inflate = 1
        self.ConvBNReLU1 = ConvBNReLU(3,32, kernel_size=3, stride=2)
        self.InvertedResidual1 = nn.Sequential(ConvBNReLU(32,32, kernel_size=3, stride=1, groups=32),
                                nn.Conv2d(32, 16, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(16))
        self.InvertedResidual2 = nn.Sequential(ConvBNReLU(16,96, kernel_size=1, stride=1),
                                ConvBNReLU(96,96, kernel_size=3, stride=2, groups=96),
                                nn.Conv2d(96, 24, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(24))
        self.InvertedResidual3 = nn.Sequential(ConvBNReLU(24,144, kernel_size=1, stride=1),
                                ConvBNReLU(144,144, kernel_size=3, stride=1, groups=144,),
                                nn.Conv2d(144, 24, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(24))
        self.InvertedResidual4 = nn.Sequential(ConvBNReLU(24,144, kernel_size=1, stride=1),
                                ConvBNReLU(144,144, kernel_size=3, stride=2, groups=144),
                                nn.Conv2d(144, 32, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(32))
        self.InvertedResidual5 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1),
                                ConvBNReLU(192,192, kernel_size=3, stride=1,groups=192),
                                nn.Conv2d(192, 32, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(32))
        self.InvertedResidual6 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1),
                                ConvBNReLU(192,192, kernel_size=3, stride=1, groups=192),
                                nn.Conv2d(192, 32, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(32))
        self.InvertedResidual7 = nn.Sequential(ConvBNReLU(32,192, kernel_size=1, stride=1),
                                ConvBNReLU(192,192, kernel_size=3, stride=2, groups=192),
                                nn.Conv2d(192, 64, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(64))
        self.InvertedResidual8 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1),
                                ConvBNReLU(384,384, kernel_size=3, stride=1, groups=384),
                                nn.Conv2d(384, 64, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(64))
        self.InvertedResidual9 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1),
                                ConvBNReLU(384,384, kernel_size=3, stride=1, groups=384),
                                nn.Conv2d(384, 64, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(64))
        self.InvertedResidual10 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1),
                                ConvBNReLU(384,384, kernel_size=3, stride=1, groups=384),
                                nn.Conv2d(384, 64, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(64))
        self.InvertedResidual11 = nn.Sequential(ConvBNReLU(64,384, kernel_size=1, stride=1),
                                ConvBNReLU(384,384, kernel_size=3, stride=1, groups=384),
                                nn.Conv2d(384, 96, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(96))
        self.InvertedResidual12 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1),
                                ConvBNReLU(576,576, kernel_size=3, stride=1, groups=576),
                                nn.Conv2d(576, 96, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(96))
        self.InvertedResidual13 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1),
                                ConvBNReLU(576,576, kernel_size=3, stride=1, groups=576),
                                nn.Conv2d(576, 96, kernel_size=1, stride=1, bias = False),
                                nn.BatchNorm2d(96))
        self.InvertedResidual14 = nn.Sequential(ConvBNReLU(96,576, kernel_size=1, stride=1),
                                ConvBNReLU(576,576, kernel_size=3, stride=2, groups=576),
                                nn.Conv2d(576, 160, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(160))
        self.InvertedResidual15 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1),
                                ConvBNReLU(960,960, kernel_size=3, stride=1, groups=960),
                                nn.Conv2d(960, 160, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(160))
        self.InvertedResidual16 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1),
                                ConvBNReLU(960,960, kernel_size=3, stride=1, groups=960),
                                nn.Conv2d(960, 160, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(160))
        self.InvertedResidual17 = nn.Sequential(ConvBNReLU(160,960, kernel_size=1, stride=1),
                                ConvBNReLU(960,960, kernel_size=3, stride=1, groups=960),
                                nn.Conv2d(960, 320, kernel_size=1, stride=1,  bias = False),
                                nn.BatchNorm2d(320))
        

        self.ConvBNReLU2 = ConvBNReLU(320, 1280, kernel_size=1, stride=1)


        # self.avgpool = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(1280,num_classes)


def mobilenet_v2(**kwargs):
    # num_classes, depth, dataset = map(kwargs.get, ['num_classes', 'depth', 'dataset'])
    # # num_classes = 1000
    return mobilenet_v2(num_classes=1000)