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

        # self.avgpool = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes, bias=False)

    def forward(self, x):
        # out = self.avgpool(out)
        out = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        out = self.dropout1(out)
        out = self.fc(out)
        return out  

def net(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    return MobileNet(num_classes=num_classes)
