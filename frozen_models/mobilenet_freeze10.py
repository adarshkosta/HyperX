import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['net']

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

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()

        # block = InvertedResidual
        norm_layer = nn.BatchNorm2d

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
        self.fc = nn.Linear(1280,num_classes, bias=False)

    def forward(self, x):
        out = self.InvertedResidual11(x)
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

def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return MobileNet(num_classes=num_classes) 