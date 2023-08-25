'''
Date: 2023-08-25 22:26:45
LastEditors: turtlepig
LastEditTime: 2023-08-26 00:25:04
Description:  
'''
import math
import time

from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import get_bleu, idx_to_word
from util.epochtimer import epoch_timer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1: # 是一个权重矩阵而非偏置项
        nn.init.kaiming_uniform(m.weight.data)

model = Transformer(
        src_pad_idx = src_pad_idx,
        trg_pad_idx = trg_pad_idx,
        trg_sos_idx = trg_sos_idx,
        enc_voc_size = enc_voc_size,
        dec_voc_size = dec_voc_size,
        d_model = d_model,
        n_head = n_heads,
        max_len = max_len,
        ffn_hidden = ffn_hidden,
        n_layers = n_layers,
        drop_prob = drop_prob,
        device = device
    ).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params = model.parameters(), lr = init_lr, weight_decay = weight_decay, eps = adam_eps)