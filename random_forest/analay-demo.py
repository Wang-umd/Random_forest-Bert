# -*- coding: utf-8 -*-
"""
@Name:analay
@author: itcast
todo: 数据分析
@Time: 2024/10/26 9:58
"""

# 1.导入依赖包
import pandas as pd
import numpy as np

import jieba
from collections import Counter

# 2.读取数据
content = pd.read_csv("data/data/train.txt", sep="\t", names=['sentence', 'label'])
# print(f'content-->{content.head()}')

# 3.统计类别数量
counter = Counter(content.label.values)
# print(counter)
# print(f'类别数量：{len(counter)}')

# 数据样本分析
# 2.1 样本总量
total = 0
for key, value in counter.items():
    total += value
print(f'样本总量：{total}')

# 2.2 类别的样本比例
for key, value in counter.items():
    print(key, (value / total) * 100, '%')

# 2.3 文本长度分析
content['sentence_len'] = content['sentence'].apply(len)
content_mean = np.mean(content['sentence_len'])
content_std = np.std(content['sentence_len'])
print(f'样本均值：{content_mean}')
print(f'样本标准差：{content_std}')


# 3.分词
# 3.1 分词
def cut_sentenc(s):
    return list(jieba.cut(s))


# 3.2 空格拼接文本，截断处理
content['words'] = content['sentence'].apply(cut_sentenc)
content['words'] = content['words'].apply(lambda s: ' '.join(s)[:30])
print(content.head())

# 3.3保存到csv
content.to_csv('./data/data/train_new.csv')
