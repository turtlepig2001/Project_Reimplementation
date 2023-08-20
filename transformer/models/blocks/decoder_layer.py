'''
Date: 2023-08-20 23:54:26
LastEditors: turtlepig
LastEditTime: 2023-08-21 00:24:31
Description:  Decoder Layer
'''
import torch.nn as nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob):
        super(DecoderLayer, self).__init__()
        # sub-layer1
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model = d_model)
        self.drop_out1 = nn.Dropout(p = drop_prob)

        # sub-layer2
        self.enc_doc_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model = d_model)
        self.drop_out2 = nn.Dropout(p = drop_prob)

        # sub-layer3
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm3 = LayerNorm(d_model = d_model)
        self.drop_out3 = nn.Dropout(p = drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q = x, k = x, v = x, mask = trg_mask)

        # 2. add and norm
        x = self.drop_out1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_doc_attention(q = x, k = enc, v = enc, mask = src_mask)
            # the output of encoder will be the key and the value

            # 4. add and norm
            x = self.drop_out2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.drop_out3(x)
        x = self.norm3(x + _x)
        
        return x

