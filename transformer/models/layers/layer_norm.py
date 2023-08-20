'''
Date: 2023-08-20 19:10:30
LastEditors: turtlepig
LastEditTime: 2023-08-20 22:21:55
Description:  layer normalization
'''
import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, esp = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.parameter.Parameter(torch.ones(d_model))
        self.beta = nn.parameter.Parameter(torch.zeros(d_model))
        self.esp = esp

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # 均值
        var = x.var(dim = -1, unbiased = False , keepdim = True) # 方差

        _x = (x - mean) / torch.sqrt(var + self.esp)
        
        out = self.gamma * _x + self.beta

        return out
    
