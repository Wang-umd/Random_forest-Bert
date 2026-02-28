# 🚀 文本多分类系统完整工程文档

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F7931E.svg?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

## 📖 项目总体简介

首先，采用经典的随机森林 (Random Forest) 算法构建多分类基线系统，为后续推荐流路由提供可靠的标签基础并提供性能评估参考。在此基础上，引入基于 Transformer 架构的预训练语言模型 BERT 作为核心引擎，通过微调 (Fine-tuning) 解决深层语义捕获等痛点，达成业务场景下的 SOTA 精度。系统还集成了模型量化压缩与 Flask Web API 部署，打通了从算法探索到工程落地的全链路。

---

## 📁 完整工程目录结构

| 模块/文件名称 | 所属子系统 | 核心功能描述 |
| :--- | :--- | :--- |
| **`data/`** | 数据层 | 包含原始训练/测试/验证集 (`train/test/dev.txt`)、10大类别映射表 (`class.txt`)、停用词表 (`stopwords.txt`) 及预处理后的 `train_new.csv`。 |
| **`analysis.py` / `analay-demo.py`** | 基线系统 | 执行探索性数据分析 (EDA) 与数据清洗。 |
| **`random_forest.py`** | 基线系统 | 包含 TF-IDF 特征提取、数据集划分与随机森林模型训练评估。 |
| **`models/bert.py`** | 深度引擎 | 模型定义与配置中心，包含 BERT 超参数配置及引入全连接层的下游分类网络架构。 |
| **`utils.py`** | 深度引擎 | 负责数据读取、`[CLS]` 符号插入、Token 转换、以及动态 Padding/Truncation 操作。 |
| **`train_eval.py`** | 深度引擎 | 封装基于 AdamW 的优化策略、交叉熵损失计算及多维度评价指标输出。 |
| **`run.py` / `run1.py`** | 深度引擎 | `run.py` 为标准微调入口；`run1.py` 为模型量化脚本，将模型在 CPU 环境下进行量化压缩。 |
| **`predict.py` / `app.py` / `demo.py`**| 服务部署 | `predict.py` 为单机推理脚本；`app.py` 基于 Flask 封装线上服务 API；`demo.py` 为 API 测试客户端。 |

---

## 🌲 核心技术点解析一：机器学习基线系统 (Random Forest)



基线模块通过对海量新闻、资讯短文本进行探索性数据分析与特征工程，能够快速、稳定地将文本划分至 10 个业务子频道。

### 1. 探索性数据分析 (EDA) 📊
* **样本均衡性分析**: 利用 `Pandas` 和 `collections.Counter` 统计 10 个类别的样本总量及占比，识别是否存在数据倾斜问题。
* **文本长度分布**: 利用 `apply(len)` 统计文本长度并计算均值与标准差，为后续特征维度设计及深度学习阶段的截断策略提供硬核的数据支撑。

### 2. 中文文本预处理与分词 ✂️
* **jieba 分词**: 采用结巴分词将连续的句子切分为词语序列，解决中文文本缺乏天然空格的特点。
* **停用词过滤**: 引入 `stopwords.txt` 剔除无实际语义贡献的标点符号、连词和虚词，大幅降低特征矩阵的噪声和稀疏度。

### 3. TF-IDF 特征工程精细调优 🧮
采用 `TfidfVectorizer` 将文本转化为向量，并进行了精细化的参数调优以提升模型表现：
* **`ngram_range=(1, 2)`**: 突破单字/单词局限，引入 2-gram 捕捉相邻词语组合形成的短语语义。
* **`max_features=10000`**: 限制最大特征维度为 1万，控制内存消耗并过滤长尾低频词以防止过拟合。
* **`min_df=2` & `max_df=0.95`**: 忽略出现少于 2 次的生僻词，并过滤在 95% 以上文档中都出现的通用废话，提升特征区分度。
* **`sublinear_tf=True`**: 采用子线性缩放（对词频进行对数缩放 $1 + \log(tf)$），削弱长文档中某个词语因重复出现次数过多产生的主导权重影响。

### 4. 模型构建与多维评估 🌲
* **分层抽样划分**: 使用 `train_test_split` 时设置 `stratify=targets`，确保训练集和测试集中各类别比例严格一致。
* **随机森林分类**: 采用基于 Bagging 思想的集成学习算法，天然具备抗过拟合能力强、对高维稀疏特征容忍度高的优势。
* **多维评估体系**: 综合采用准确率 (`accuracy_score`)、精确率 (`precision_score`)、召回率 (`recall_score`) 和 F1 值 (`f1_score`) 进行全面评估，并通过交叉验证 (`cross_val_score`) 保证模型稳定性。

---

## 🚀 核心技术点解析二：深度学习 SOTA 引擎 (BERT)



在基线的基础上，深度学习模块引入 BERT 模型，有效解决了传统机器学习无法捕获深层上下文语义、多义词及长距离依赖的痛点。

### 1. BERT 深度语义表征 
* **双向 Transformer 编码**: 采用 `transformers` 库加载预训练 BERT 模型，提取输入文本的 768 维密集型向量表示，精准捕获字词在特定语境下的真实含义。
* **`[CLS]` 聚合分类**: 在每条新闻短文本首部拼接 `[CLS]` 标记，利用该位置输出的 Hidden State 作为全局语义表征，直接接入全连接层完成 10 分类任务。

### 2. 精细化的数据构造与 Mask 机制
* **定长截断与填充**: 统一序列长度为 32 (`pad_size=32`)，采用短填长切策略，保证 GPU Batch 并行计算效率。
* **Attention Mask**: 构造与文本长度对应的 `mask` 张量（真实文本处为 1，Padding 处为 0），防止模型将注意力浪费在无意义的补全符上。

### 3. 高级优化策略 (AdamW & 权重衰减)
* **差分权重衰减**: 将模型参数分组，对 `bias` 和 `LayerNorm` 的参数不应用 L2 正则化（`weight_decay=0.0`），而对其他权重应用 `weight_decay=0.01`。这种细粒度的正则化控制有效防止了模型过拟合，是微调的最佳实践。

### 4. 工业级模型量化优化 (Quantization)
* **动态量化**: 利用 PyTorch 动态量化技术，将模型中 FP32（32位浮点数）的权重转换为 INT8（8位整数）。
* **性能突围**: 在几乎不损失精度的前提下，成倍降低了内存/显存占用，并显著加快了 CPU 环境下的推理速度，契合了推荐系统对延迟的严苛要求。

### 5. 在线服务化部署 (Flask API)
* **工程化服务封装**: 提供标准的 HTTP POST 接口，接收 JSON 格式数据。
* **开箱即用**: 内部包含完整的分词、Tensor 转换、模型前向传播推理及 ID-类别名映射，真正实现了算法能力的在线服务化落地。
