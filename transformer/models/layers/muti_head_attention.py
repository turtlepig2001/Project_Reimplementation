'''
Date: 2023-08-20 00:16:26
LastEditors: turtlepig
LastEditTime: 2023-08-20 19:08:28
Description:  Multi-Head Attention
'''
import torch.nn as nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MulitHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head):
        super(MulitHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        # 本质是一组映射权重
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):

        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads


    def split(self, tensor):
        r"""
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor] | d_tensor = d_model / n_head
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):
        r"""
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, d_model)
        
        return tensor
