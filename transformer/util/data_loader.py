'''
Date: 2023-08-18 23:41:27
LastEditors: turtlepig
LastEditTime: 2023-08-19 00:33:14
Description:  DataLoader
'''
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

# for the torchtext version of 0.6.0, we need to use the two lines above instead of the next two lines

# from torchtext.legacy.data import Field, BucketIterator
# from torchtext.legacy.datasets.translation import Multi30k

class DataLoader:
    source:Field = None
    target:Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        r'''
        Parameters:
            ext: 文件拓展名.
            tokenize_en: 分词函数 对应英语
            tokenize_de: 分词函数 对应德语
            init_token/eos_token: 起始和结束的特殊标记
        '''
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')
    
    def make_dataset(self):
        r'''
        
        '''
        if self.ext == ('.de', '.en'):
            # 源语言为德语 目标语言英语
            self.source = Field(tokenize = self.tokenize_de, init_token = self.init_token, eos_token = self.eos_token, lower = True, batch_first = True)
            
            self.target = Field(tokenize = self.tokenize_en, init_token = self.init_token, eos_token = self.eos_token, lower = True, batch_first = True)

        elif self.ext == ('.en', 'de'):
            # 源语言为英语 目标语言德语
            self.source = Field(tokenize = self.tokenize_en, init_token = self.init_token, eos_token = self.eos_token, lower = True, batch_first = True)

            self.target = Field(tokenize = self.tokenize_de, init_token = self.init_token, eos_token = self.eos_token, lower = True, batch_first = True)

        train_data, valid_data, test_data = Multi30k.splits(exts = self.exts, fields = (self.source, self.target) )
        
        return train_data, valid_data, test_data
    
    def build_vocab(self, train_data, min_freq):
        r'''
        构造词汇表
        Parameters:
            min_freq: 单词出现的最小频率
        '''
        self.source.build_vocab(train_data, min_freq)
        self.target.build_vocab(train_data, min_freq)

    def make_iter(self, train, valid, test, batch_size, device):
        r'''
        创建数据加载器，划分批次
        '''
        train_iterator , valid_iterator, test_iterator = BucketIterator((train, valid, test), batch_size = batch_size, device = device)

        print('dataset initializing done')

        return train_iterator, valid_iterator, test_iterator





        
