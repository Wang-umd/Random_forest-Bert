"""
## 数据集分析  analysis.py
# 1.导入依赖包
# 2.读取数据
    # 2.1 打印前10行
    # 2.2 获取样本数量
    # 2.3 统计每个类别的数量
    # 2.4 打印信息
# 3.统计样本总量
    # 3.1 遍历每个类别的样本数量
    # 3.2 计算每个类别样本数量的比例
    # 3.3 绘制图表
    # 3.4 统计每行样本的长度
    # 3.5 统计文本长度的均值和方差
# 4.结巴分词
    # 4.1 添加一列数据存储分词后的结果
    # 4.2 分词
    # 4.3 将分词后的结果只保留30个元素
    # 4.4 结果存放到csv文件中
"""

# 1.导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import jieba

# 2.读取数据
content = pd.read_csv('./data/train.txt', sep='\t')
# 2.1 打印前10行
print(content.head(10))
# 2.2 获取样本数量
print(len(content))
# 2.3 统计每个类别的数量
count = Counter(content.label.values)
# 2.4 打印信息
print(count)
print(len(count))
# print(content.columns)
print('***************************************')

# 3.统计样本总量
values = []
ratios = []
total = 0
# 3.1 遍历每个类别的样本数量
for i, v in count.items():
    total += v
    values.append(v)
print(total)
# 3.2 计算每个类别样本数量的比例
for i, v in count.items():
    ratio = v / total * 100
    ratios.append(ratio)
    print(i, ratio, '%')

# # 3.3 绘制图表
# plt.bar(range(10), values)
# plt.show()
# plt.pie(ratios, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.title('样本类别比例图')
# plt.show()

print('***************************************')
# 3.4 统计每行样本的长度
content['sentence_len'] = content['sentence'].apply(len)
print(content.head(10))
# 3.5 统计文本长度的均值和方差
length_mean = np.mean(content['sentence_len'])
length_std = np.std(content['sentence_len'])
print('length_mean = ', length_mean)
print('length_std = ', length_std)


# 4.结巴分词
def cut_sentence(s):
    return list(jieba.cut(s))


# 4.1 添加一列数据存储分词后的结果
content['words'] = content['sentence'].apply(cut_sentence)
# 打印前十行
print(content.head(10))
# 4.2 分词
content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))

# 4.3 将分词后的结果只保留30个元素
content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:30])
# 4.4 结果存放到csv文件中
content.to_csv('./data/train_new.csv')
