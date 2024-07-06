import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from swifter import swifter

import util
import sentiment_dataset as sed


EN_BERT_PATH = './data/bert-base-uncased'
IMDB_FILE = './data/IMDB Dataset.csv'


class BertTransformerEncoder(nn.Module):
    def __init__(self, n_classes, nhead=8, d_model=768, num_layers=6):
        super().__init__()
        self.bert = BertModel.from_pretrained(EN_BERT_PATH)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        # 用 bert 代替 nn.Embedding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden_states = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # transformer encoder -> pooling -> dropout -> fully_connected_layer
        encoded_output = self.transformer_encoder(encoder_hidden_states)
        # 通过池化获得句子级别的表征
        pooled_output = encoded_output.mean(dim=1)
        # 通过 dropout 防止过拟合
        x = self.drop(pooled_output)
        # 最后接一个全连接层
        return self.out(x)


def prepare(sample_num=None, df=None):
    if df is not None:
        df = pd.read_csv(IMDB_FILE)
        if sample_num is not None:
            df = df.sample(n=sample_num)

    # 去除 html 标签
    def remove_html_label(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    X = df['review'].swifter.apply(remove_html_label).tolist()
    y = df['sentiment'].swifter.apply(lambda e: 1 if e == 'positive' else 0).tolist()
    
    return X, y
    

def load_data(X, y, batch_size):
    # 分割训练集和测试集（其实是验证集）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

    tokenizer = BertTokenizer.from_pretrained(EN_BERT_PATH)

    train_dataset = sed.SentimentDataset(X_train, y_train, tokenizer, max_len=512)
    test_dataset = sed.SentimentDataset(X_test, y_test, tokenizer, max_len=512)
    
    # 用 DataLoader 加载数据
    train_iter = DataLoader(train_dataset,
                            batch_size,
                            shuffle=True,
                            num_workers=4)
    test_iter = DataLoader(test_dataset,
                           batch_size,
                           shuffle=True,
                           num_workers=4)
    return train_iter, test_iter


def evaluate_accuracy(net, data_iter, device):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = util.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for data in data_iter:
            y_hat = net(input_ids=data['input_ids'].to(device),
                        attention_mask=data['attention_mask'].to(device))
            y = data['label'].to(device)
            metric.add(util.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater, device):
    """训练模型一个迭代周期"""

    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()

    # 训练损失总和、训练准确度总和、样本数
    metric = util.Accumulator(3)
    for data in train_iter:
        # 计算梯度并更新参数
        y_hat = net(input_ids=data['input_ids'].to(device),
                      attention_mask=data['attention_mask'].to(device))

        y = data['label'].to(device)
        l = loss(y_hat, y)

        # 使用PyTorch内置的优化器和损失函数
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        metric.add(float(l.sum()), util.accuracy(y_hat, y), y.numel())

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater, device):
    """训练模型"""
    # animator = util.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1.0],
    #                     legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        print(f'epoch: {epoch}')
        train_metrics = train_epoch(net, train_iter, loss, updater, device)
        test_acc = evaluate_accuracy(net, test_iter, device)
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

    print(f'train_loss: {train_loss:.3f}')
    print(f'train_acc: {train_acc:.3f}')
    print(f'test_acc: {test_acc:.3f}')


def main(batch_size, num_epochs, sample_num, df=None):
    X, y = prepare(sample_num, df)
    train_iter, test_iter = load_data(X, y, batch_size)

    model = BertTransformerEncoder(n_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 优化器
    optimizer = torch.optim.Adam(model.parameters())

    # 训练模型
    train(net=model,
      train_iter=train_iter,
      test_iter=test_iter,
      loss=loss,
      num_epochs=num_epochs,
      updater=optimizer,
      device=device)


if __name__ == "__main__":
    main(batch_size=256,
         num_epochs=10,
         sample_num=20)
