import json
import numpy as np
import pandas as pd
import torch
from ast import literal_eval


# 参数 ebd_cols 定义哪些列存了 embedding
def embedding_df_to_csv(df, csv_path, ebd_cols: list):
    """将带有 embedding 的 DataFrame 存入 csv"""
    def ebd2str(embedding):
        if not isinstance(embedding, list):
            ebd = embedding.tolist()
        return json.dumps(ebd)

    for col in ebd_cols:
        df[col] = df[col].apply(ebd2str)

    df.to_csv(csv_path, index=False)


def read_embedding_csv(csv_path, ebd_cols: list):
    """将带有 embedding 的 csv 读入 DataFrame"""
    df = pd.read_csv(csv_path)
    for col in ebd_cols:
        df[col] = df[col].apply(literal_eval).apply(lambda e: np.array(e))

    return df


def get_avg_embeddings(sentences, tokenizer, model):
    """计算句子的平均嵌入"""
    encoded_inputs = tokenizer(corpus, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings
