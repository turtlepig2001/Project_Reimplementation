'''
Date: 2023-08-19 23:18:08
LastEditors: turtlepig
LastEditTime: 2023-08-20 00:05:37
Description:  Final transformer embedding
'''

import torch.nn as nn

from embedding.positional_encoding import PositionalEncoding
from embedding.token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    r"""
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_out, device):
        r"""
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p = drop_out)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)

        return self.drop_out(tok_emb, pos_emb)