'''
Date: 2023-08-20 00:24:06
LastEditors: turtlepig
LastEditTime: 2023-08-22 18:11:48
Description:  Scale Dot Product Attention
'''
import math

import torch.nn as nn

class ScaleDotProductAttention(nn.Module):
    r"""
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v , mask =None, e = 1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]

        bath_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3) # 下标从0开始
        score = (q @ k_t) / math.sqrt(d_tensor) # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score