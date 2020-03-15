import numpy as np
import torch
import torch.nn.functional as F

from ...tools import torchutils as tu
from .attention import Attention

class SelfAttention(Attention):
    '''
        Self attention mechanism based on the paper: Attention is all you need (https://arxiv.org/abs/1706.03762).
    '''

    def __init__(self, input_shape, output_shape):
        super(SelfAttention, self).__init__(input_shape, output_shape)


    def forward(self, x):
        '''
            Run input x through this self attention mechanism.
            Arguments:
                x: a batch of sentances (N x S x W) where N is the batch size, S is sentance size, W is the word size (input_shape)
            Returns:
                a N x Z x Z tensor with an attention vector for each word in each sentance
        '''
        return self.b_attention(x, x, x)
    
class MultiHeadSelfAttention(torch.nn.Module):
    '''
        Multi-head version of self attention mechanism based on the paper: Attention is all you need (https://arxiv.org/abs/1706.03762).
    '''

    def __init__(self, input_shape, output_shape, heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert heads > 1 #otherwise consider using a single SelfAttention
        self.heads = []
        
        for i in range(heads):
            head = SelfAttention(input_shape, output_shape)
            self.heads.append(head)
            self.add_module("head-{0}".format(i), head)
        self.WZ = torch.nn.Parameter(torch.randn(output_shape, heads * output_shape))
    
    def forward(self, x):
        z = torch.cat([head(x) for head in self.heads], dim=-1)
        return z @ self.WZ.T
