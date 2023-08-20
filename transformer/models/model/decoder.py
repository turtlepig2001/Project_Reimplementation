'''
Date: 2023-08-21 00:27:51
LastEditors: turtlepig
LastEditTime: 2023-08-21 00:59:40
Description:  complete decoder
'''
import torch.nn as nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size,max_len, d_model, n_head, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(vocab_size = dec_voc_size, d_model = d_model, max_len = max_len, drop_prob = drop_prob, device = device)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model, n_head = n_head, ffn_hidden = ffn_hidden, drop_prob = drop_prob) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size) #

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.linear(trg)
        return output