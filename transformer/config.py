'''
Date: 2023-08-19 15:46:59
LastEditors: turtlepig
LastEditTime: 2023-08-19 16:02:39
Description:  configuration settings
'''
import torch

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# parameters setting
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameters setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10 #早停耐心值
warmup = 100
epoch = 1000
clip = 1.0 # 梯度裁剪阈值
weight_decay = 5e-4
inf = float('inf')