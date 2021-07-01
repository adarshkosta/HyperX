import torch
import torch.nn as nn
import pdb

def float_to_16bits_tensor_fast(input, frac_bits, bit_slice, bit_slice_num, input_bits): # input is batch x n tensor / output is n x 16 tensor..
    int_bit = input_bits - frac_bits -1 # extra -1 for sign bit
    #clamp data into the available range
    input = torch.clamp(input, -2**int_bit, 2**int_bit-1/2**frac_bits)
    #normalize
    input = input.div_(2**int_bit)
    #left shift all the fracional values so that all 16 bits comes to the left of decimal
    input = input.mul_(2**(input_bits-1))
    #take the integer part of the input, which represents our 16bit number
    input = torch.floor(input)
    #divide by scalar to get the decimal representation back, MSB----->LSB
    input_sliced = torch.stack([torch.floor(torch.div(input, 2**(i*bit_slice))) - \
                                torch.mul(torch.floor(torch.div(input, 2**((i+1)*bit_slice))), 2**bit_slice) for i in range(bit_slice_num-1,-1,-1) ])
    # pdb.set_trace()
    del input
    return input_sliced.permute(1,2,0)

def act_dquant(inp, B, I):
    F = B-I
    # act = torch.clamp(inp, 1/n, 1-1/n)

    act = torch.clamp(inp, -2**I, 2**I-1/2**F)

    act = act/2**I

    return (torch.floor(act*2**B)/2**B)*2**I

def wt_dquant(inp, B, I):
    F = B-I
    # wt = torch.tanh(inp)
    wt = inp

    # wt = wt/2 + 0.5
    # wt = torch.clamp(wt, 1/2**B, 2**I-1/2**B)
    # return (2*(torch.floor(wt*2**B)/2**B) - 1)

    wt = torch.clamp(wt, -2**I, 2**I-1/2**F)

    wt = wt/2**I

    return (torch.floor(wt*2**B)/2**B)*2**I


def act_fsquant(inp, B, I):
    S = 1
    F = B-I-S

    return (torch.floor((torch.clamp(inp, -2**I, 2**I-1/2**F)/2**I)*2**(B-1))/2**(B-1))*2**I

def wt_fsquant(inp, B, I):
    S = 1
    F = B-I-S

    return (torch.floor((torch.clamp(inp, -2**I, 2**I-1/2**F)/2**I)*2**(B-1))/2**(B-1))*2**I


W = 20*torch.rand(1,100000) - 10
A = 20*torch.rand(1,100000) - 10
B = 8
I = 2

print('Weight range: ({}, {})'.format(W.min(), W.max()))
print('Activation range: ({}, {})'.format(A.min(), A.max()))

print('D-Quant:',torch.unique(act_dquant(inp=A, B=7, I=I)))
print(len(torch.unique(act_dquant(inp=A, B=7, I=I))))

print('FS-Quant:', torch.unique(act_fsquant(inp=A, B=B, I=I)))
print(len(torch.unique(act_fsquant(inp=A, B=B, I=I))))


print('D-Quant:',torch.unique(wt_dquant(inp=W, B=7, I=I)))
print(len(torch.unique(wt_dquant(inp=W, B=7, I=I))))

print('FS-Quant:',torch.unique(wt_fsquant(inp=W, B=B, I=I)))
print(len(torch.unique(wt_fsquant(inp=W, B=B, I=I))))

fptfxp = float_to_16bits_tensor_fast(A, 7, 1, 8, 8)


