# sentiment-analysis

> 用 Bert + Transformer Encoder + MLP 做文本情感分析。

- IMDB 数据集：<a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data" target="_blank">imdb-dataset</a>

本文要点：

1. 建立 词向量 ⇋ CSV 文件 双向 Pipeline
2. 用两种方法对 IMDB 电影评论做情感分析：
    - Bert 预训练词向量 + MLP
    - Bert + Transformer Encoder + 全连接层

✨ PS: 前两章是 Pipeline 代码，建议从第三章看起。

### 一、读写词向量

本节的主要目标是完成 `词向量 -> CSV 文件` 和 `CSV 文件 -> 词向量` 的 Pipeline。

1. 对语料做预处理
2. 获取词向量和句子向量
3. 将词向量存入 csv
4. 从 csv 中读取词向量
5. 将读写词向量功能整合成函数


### 二、获取 IMDB 数据集的 Embedding

将 IMDB 数据集中的电影评论转换成句子向量，然后存在 CSV 文件中。

1. 文本预处理
2. 计算句子向量


### 三、用 MLP 做文本情感分析

用 Bert + MLP 做 IMDB 电影评论文本情感分析。

1. 从 csv 读入 embedding
2. 定义 MLP
3. 训练模型
4. 预测


### 四、用 Transformer 做文本情感分析

用 Bert + Transformer Encoder + MLP 做 IMDB 电影评论文本情感分析。

1. 数据预处理
2. 加载数据集
3. 定义模型
4. 训练模型


### 五、模块调用

将第 4 节代码整合成 `transformer_encoder.py` 文件，再调用此文件做推理或训练模型。

1. 推理
2. 训练
