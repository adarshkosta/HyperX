import torch
import torch.nn as nn
from collections import namedtuple

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def quantize(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = x.min()

    q_x = zero_point + ((x-zero_point)/scale).round_()*scale

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


# t = torch.rand(5,5)
# qt = quantize_tensor(t, num_bits=8)
# print(t)
# print(qt)
# print(qt.scale*qt.tensor.float())