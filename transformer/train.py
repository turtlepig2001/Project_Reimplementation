'''
Date: 2023-08-25 22:26:45
LastEditors: turtlepig
LastEditTime: 2023-08-26 23:43:37
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

# 学习率调整
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, verbose = True, factor = factor, patience = patience)

# torch:Specifies a target value that is ignored and does not contribute to the input gradient
criterion = nn.CrossEntropyLoss(ignore_index = src_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train() # 进入训练模式
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg[:, : -1]) # why -1: avoid '<sos>'

        # output : batch_size x seq_len 
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, : -1].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        # 梯度剪裁
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval() # 进入评估模式
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, : -1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, : -1].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim = 1)[1] # ????
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypothesis = output_words.split(), reference = trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu