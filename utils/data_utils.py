"""
数据处理工具函数
"""
import pandas as pd
from typing import List


def preprocess_data(data_path: str, dataset_type: str = "politifact") -> pd.DataFrame:
    """
    数据预处理
    
    Args:
        data_path: 数据文件路径
        dataset_type: 数据集类型
        
    Returns:
        预处理后的 DataFrame
    """
    df = pd.read_pickle(data_path)
    
    # 文本清理
    df["text"] = df["text"].str.replace(r"[^\w\s\u4e00-\u9fa5]", "", regex=True)
    df["text"] = df["text"].str.strip()
    
    # 数据验证
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= 10]
    
    # 标签平衡
    if "label" in df.columns:
        min_count = df["label"].value_counts().min()
        df_balanced = df.groupby("label").head(min_count).reset_index(drop=True)
        return df_balanced
    return df

