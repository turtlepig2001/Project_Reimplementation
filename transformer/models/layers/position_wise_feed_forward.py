'''
Date: 2023-08-20 22:27:01
LastEditors: turtlepig
LastEditTime: 2023-08-20 22:32:29
Description:  position wise feed forward
'''
import torch.nn as nn

# apply to each position

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x 