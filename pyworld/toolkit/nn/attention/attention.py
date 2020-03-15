
import numpy as np
import torch
import torch.nn.functional as F

from ...tools import torchutils as tu

class Attention(torch.nn.Module):
    '''
        Self attention mechanism based on the paper: Attention is all you need (https://arxiv.org/abs/1706.03762).
    '''

    def __init__(self, input_shape, output_shape):
        '''
            Constructor.
            Arguments:
                input_shape: size of an input word
                output_shape: size of attention encoding
        '''
        super(Attention, self).__init__()
        input_shape = tu.as_shape(input_shape)
        output_shape = tu.as_shape(output_shape)
        self.WV = torch.nn.Parameter(torch.randn(output_shape[0], input_shape[0]))
        self.WQ = torch.nn.Parameter(torch.randn(output_shape[0], input_shape[0]))
        self.WK = torch.nn.Parameter(torch.randn(output_shape[0], input_shape[0]))
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, vkq):
        v,k,q = vkq
        '''
            Run input x through this self attention mechanism.
            Arguments:
                x: a tuple (v,k,q)
                v: a batch of sentences (N x S x W) where N is the batch size, S is sentance size, W is the word size (input_shape)
                k: TODO
                q: TODO
            Returns:
                a N x Z x Z tensor with an attention vector for each word in each sentance
        '''
        return self.b_attention(v, k, q)

    def attention_mask(self, _, k, q): #no batch, used for visualisation
        '''
            Compute the attention mask - this will give a 2D heatmap with each value associated to a pairs of words in the sentance.
        '''
        wk = k @ self.WK.T
        wq = q @ self.WQ.T
        return torch.nn.functional.softmax(wq @ wk.T, dim=1) / self.input_shape[0]
    
    def b_attention_mask(self, _, k, q):
        '''
            Batched version of attention_mask.
        '''
        wk = k @ self.WK.T
        wq = q @ self.WQ.T
        return torch.nn.functional.softmax(torch.bmm(wq, wk.permute(0,2,1)), dim=1) / self.input_shape[0]
          
    def attention(self, v, k, q): #no batch
        wv = v @ self.WV.T
        wk = k @ self.WK.T
        wq = q @ self.WQ.T
        wqk = torch.nn.functional.softmax(wq @ wk.T, dim=1) / self.input_shape[0]
        return wqk @ wv
    
    def b_attention(self, v, k, q): #batch
        wv = v @ self.WV.T
        wk = k @ self.WK.T
        wq = q @ self.WQ.T
        wqk = torch.nn.functional.softmax(torch.bmm(wq, wk.permute(0,2,1)), dim=1) / self.input_shape[0]
        return torch.bmm(wqk, wv)