'''
Date: 2023-08-19 00:43:38
LastEditors: turtlepig
LastEditTime: 2023-08-19 13:27:29
Description:  bleu compute
'''
import math
from collections import Counter
import numpy as np

def bleu_states(hypothesis, reference):
    r"""
    Compute statistics for BLEU.
    """
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    
    for n in range(1,5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )

        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]    
        )

        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        # 计算交集中的n-gram总和，共同出现的 n-gram 数量小于零，会被调整为零
        
        stats.append(max([len(hypothesis) + 1 -n, 0]))
        # n-gram 的总数

    return stats

def bleu(stats):
    r"""
    Compute BLEU given n-gram statistics.
    """
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    #存在一个以上的 n-gram 统计为 0 表示翻译无法匹配参考

    (c, r) = stats[ : 2]
    log_bleu_prec = sum(
        [math.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])]    
    ) / 4

    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)

def get_bleu(hypothesis, reference):
    r"""
    Get validation BLEU score for dev set.
    """
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    for hyp, ref in zip(hypothesis, reference):
        stats += np.array(bleu_states(hyp, ref))
    
    return 100 * bleu(stats)

def idx_to_word(x, vocab):
    r"""
    将索引序列转换为单词序列的函数
    """
    words = []
    for i in x:
        word = vocab.itos[i]
        if ' < ' not in word: #过滤特殊标记
            words.append(word)
    words = " ".join(words) # 用空格隔开
    return words