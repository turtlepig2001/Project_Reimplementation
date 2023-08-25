'''
Date: 2023-08-25 22:26:35
LastEditors: turtlepig
LastEditTime: 2023-08-25 22:59:49
Description:  data
'''
from config import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(
        ext = ('.en', '.de'),
        tokenize_en = Tokenizer.tokenize_en,
        tokenize_de = Tokenizer.tokenize_de,
        init_token = '<sos>',
        eos_token = '<eos>'
    )

train, valid, test = loader.make_dataset()

loader.build_vocab(train_data = train, min_freq = 2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test, batch_size, device)

# 获得词表中对应词的索引
src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)