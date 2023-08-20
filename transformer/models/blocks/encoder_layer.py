'''
Date: 2023-08-20 22:35:02
LastEditors: turtlepig
LastEditTime: 2023-08-20 23:56:49
Description:  Encoder Layer
'''
import torch.nn as nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, n_head = n_head)
        self.norm1 = LayerNorm(d_model = d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)

        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNorm(d_model = d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)
    
    def forward(self, x, src_mask):

        _x = x
        # 1. compute self attention
        x = self.attention(q = x, k = x, v = x, mask = src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x