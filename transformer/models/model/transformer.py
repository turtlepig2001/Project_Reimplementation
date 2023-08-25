'''
Date: 2023-08-21 00:28:08
LastEditors: turtlepig
LastEditTime: 2023-08-22 18:10:26
Description:  Whole Transformer
'''

import torch
import torch.nn as nn

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx # 源序列填充标记的索引
        self.trg_pad_idx = trg_pad_idx # 目标序列填充标记的索引
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size = enc_voc_size, max_len = max_len, d_model = d_model, n_head = n_head, ffn_hidden = ffn_hidden,n_layers = n_layers, drop_prob = drop_prob, device = device)

        self.decoder = Decoder(dec_voc_size = dec_voc_size, max_len = max_len, d_model = d_model, n_head = n_head, ffn_hidden = ffn_hidden, drop_prob = drop_prob, device = device)

    def forward(self, src, trg):
        r"""
        
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output

    def make_src_mask(self, src):
        r"""
        Parameters:
        src: batch_size x seq_len
        """
        # 避免当前位置的信息与填充位置（pad positions）的信息进行计算，需要将填充位置的信息掩盖掉

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x 1 x seq_len
        # src_atten_score: batch_size x heads x seq_len x seq_len 
        return src_mask
    
    def make_trg_mask(self, trg):
        # 确保在预测目标序列时不会使用未来位置的信息

        # initial input batchsize x seq_len after the embeddin : batch_size x seq_len x d

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x trg_seq_len x 1
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        # troch.tril生成一个下三角矩阵(0/1) 用作自回归掩码 
        
        trg_mask = trg_pad_mask & trg_sub_mask
        # batch_size x 1 x trg_seq_len x trg_seq_len
        return trg_mask