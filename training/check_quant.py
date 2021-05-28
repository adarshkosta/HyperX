import torch
import torch.nn as nn

def dquant(A, B):
    n = 2**B - 1

    return torch.round(torch.clamp(A, 0, 1)*n)/n

def fsquant(A, B, F):
    S = 1
    I = B-F-S

    return torch.floor((torch.clamp(A, -2**I, 2**I-1/2**F)/2**I)*2**(B-1))


A = torch.tensor(0.68)
B = 8

print(dquant(A=A, B=B)*255)

print(fsquant(A=A, B=B, F=7))
print(fsquant(A=A, B=B, F=6))
print(fsquant(A=A, B=B, F=5))