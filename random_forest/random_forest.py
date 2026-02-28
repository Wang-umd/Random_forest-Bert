"""
## 数据集分析  random_forest.py
# 1.导入依赖包
# 2.读取数据集
# 3.构建语料库
# 4.获取停用词
# 5.计算tfidf特征stopwords.txt
# 6.划分数据集
# 7.实例化模型
# 8.模型训练
# 9.模型评估
"""

# 1.导入依赖包
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

# 2.读取数据集
# 指定数据集的位置
TRAIN_CORPUS = './data/train_new.csv'
STOP_WORDS = './data/stopwords.txt'
WORDS_COLUMN = 'words'

content = pd.read_csv(TRAIN_CORPUS)

# 3.构建语料库
corpus = content[WORDS_COLUMN].values

# 4.获取停用词
stop_words = open(STOP_WORDS, encoding='utf-8').read().split()


# 5.计算tfidf特征stopwords.txt
#tfidf = TfidfVectorizer(stop_words=stop_words)
# TF-IDF参数调优
# 增加n-gram范围(1,2)捕捉词语组合
# 设置max_features限制特征维度
# 添加min_df/max_df过滤极端词频词汇
# 使用sublinear_tf改善长文档表现
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    max_features=10000,        # 限制特征数量
    ngram_range=(1, 2),        # 使用1-gram和2-gram
    min_df=2,                  # 忽略出现次数少于2的词
    max_df=0.95,               # 忽略出现在95%以上文档中的词
    sublinear_tf=True          # 使用子线性缩放
)

text_vectors = tfidf.fit_transform(corpus)
# print(tfidf.vocabulary_)
# print(text_vectors)
# 目标值
targets = content['label']


# 6.划分数据集
x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=22, stratify=targets)

#网格搜索最优参数
#使用GridSearchCV自动寻找最优参数组合
#调整n_estimators、max_depth、min_samples_split等关键参数
param_grid = {
    'n_estimators': [100, 200],      # 2种
    'max_depth': [10, 20, None],     # 3种
    'min_samples_split': [2, 5]
}

# 7.实例化模型
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=22, n_jobs=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
rf_grid.fit(x_train, y_train)

model = rf_grid.best_estimator_

# 8.模型训练
model.fit(x_train, y_train)

# 9.模型评估
accuracy = accuracy_score(model.predict(x_test), y_test)
ic(accuracy)
