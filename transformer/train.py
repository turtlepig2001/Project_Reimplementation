'''
Date: 2023-08-25 22:26:45
LastEditors: turtlepig
LastEditTime: 2023-09-02 23:15:18
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
        nn.init.kaiming_uniform(m.weight.data) # 初始化权重

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

print(f'The model has {count_parameters(model):,} trainable parameters') # 千分位分隔
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
                    output_words = output[j].max(dim = 1)[1] # max returns tensor and indices, we need indices
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypothesis = output_words.split(), reference = trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)  # 学习率调整

        train_losses.append(train_loss)
        test_losses.append(valid_loss)

        epoch_mins, epoch_secs = epoch_timer(start_time, end_time)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model_{0}.pt'.format(valid_loss)) # 0是占位符

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step+1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(total_epoch = epoch, best_loss = inf)