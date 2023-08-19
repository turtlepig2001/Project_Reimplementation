'''
Date: 2023-08-19 23:16:16
LastEditors: turtlepig
LastEditTime: 2023-08-19 23:16:17
Description:  token embedding
'''

from torch import nn


class TokenEmbedding(nn.Embedding):
    r"""
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        r"""
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)