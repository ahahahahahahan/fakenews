"""
评估指标计算工具函数
"""
from typing import List, Dict
from sklearn.metrics import f1_score


def normalize_label(label: str) -> int:
    """
    将标签标准化为 0 或 1
    
    Args:
        label: 原始标签
        
    Returns:
        标准化后的标签（0=真实，1=虚假）
    """
    label_str = str(label).lower()
    if label_str in ["虚假", "false", "1", "fake"]:
        return 1
    elif label_str in ["真实", "true", "0", "real"]:
        return 0
    return None


def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        
    Returns:
        包含各种评估指标的字典
    """
    # 统一标签格式
    true_labels_processed = []
    pred_labels_processed = []
    
    for t, p in zip(true_labels, pred_labels):
        # 处理真实标签
        t_normalized = normalize_label(t)
        if t_normalized is None:
            continue
        
        # 处理预测标签
        p_normalized = normalize_label(p)
        if p_normalized is None:
            # 如果无法识别，假设为错误预测
            p_normalized = 1 - t_normalized
        
        true_labels_processed.append(t_normalized)
        pred_labels_processed.append(p_normalized)
    
    if len(true_labels_processed) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "macro_f1": 0.0}
    
    # 计算混淆矩阵
    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels_processed, pred_labels_processed))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels_processed, pred_labels_processed))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels_processed, pred_labels_processed))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels_processed, pred_labels_processed))
    
    # 计算指标
    total = len(true_labels_processed)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    macro_f1 = f1_score(true_labels_processed, pred_labels_processed, average='macro')
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "macro_f1": round(macro_f1, 3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

