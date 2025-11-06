"""
文本嵌入计算工具函数
"""
import numpy as np
from typing import List
from numpy.linalg import norm
import copy

# 尝试导入 sentence-transformers，如果失败则使用简单的 TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except ImportError:
    print("警告: sentence-transformers 未安装，将使用简单的文本相似度方法")
    USE_SBERT = False
    from sklearn.feature_extraction.text import TfidfVectorizer

from config import SSR_EMBEDDING_MODEL, SSR_TFIDF_MAX_FEATURES


class EmbeddingComputer:
    """文本嵌入计算器"""
    
    def __init__(self, use_sbert: bool = None):
        """
        初始化嵌入计算器
        
        Args:
            use_sbert: 是否使用 sentence-transformers，None 时自动检测
        """
        if use_sbert is None:
            use_sbert = USE_SBERT
        
        self.use_sbert = use_sbert
        if use_sbert:
            print("加载 sentence-transformers 模型用于 SSR...")
            self.embedding_model = SentenceTransformer(SSR_EMBEDDING_MODEL)
        else:
            print("使用 TF-IDF 进行文本相似度计算...")
            self.embedding_model = None
            self.tfidf_vectorizer = TfidfVectorizer(max_features=SSR_TFIDF_MAX_FEATURES)
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        计算文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量数组
        """
        if self.use_sbert and self.embedding_model:
            return self.embedding_model.encode(texts, show_progress_bar=False)
        else:
            # 使用 TF-IDF
            vectors = self.tfidf_vectorizer.fit_transform(texts)
            return vectors.toarray()


def compute_similarity_scores(test_embeddings: np.ndarray, train_embeddings: np.ndarray) -> np.ndarray:
    """
    计算相似度分数矩阵
    
    Args:
        test_embeddings: 测试数据嵌入向量
        train_embeddings: 训练数据嵌入向量
        
    Returns:
        相似度分数矩阵 (test_size, train_size)
    """
    similarity_scores = np.zeros((len(test_embeddings), len(train_embeddings)))
    
    dot_products = np.dot(test_embeddings, train_embeddings.T)
    norms_test = norm(test_embeddings, axis=1, keepdims=True)
    norms_train = norm(train_embeddings, axis=1, keepdims=True).T
    
    norms_test[norms_test == 0] = 1
    norms_train[norms_train == 0] = 1
    similarity_scores = dot_products / (norms_test * norms_train)
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)
    similarity_scores[similarity_scores >= 1.0] = 0.0
    
    return similarity_scores


def extract_top_k_similar(similarity_scores: np.ndarray, k: int) -> List[Dict]:
    """
    提取 top-k 相似样本
    
    Args:
        similarity_scores: 相似度分数矩阵
        k: 提取的相似样本数量
        
    Returns:
        每个测试样本的 top-k 相似样本列表
    """
    similarity_scores_copy = copy.deepcopy(similarity_scores)
    results = []
    
    for i in range(len(similarity_scores)):
        samples = []
        scores = []
        current_scores = similarity_scores_copy[i]
        
        for _ in range(k):
            if np.max(current_scores) <= 0:
                break
            j = int(np.argmax(current_scores))
            samples.append(j)
            scores.append(float(current_scores[j]))
            current_scores[j] = -1
        
        results.append({
            "index": i,
            "samples": samples,  # 训练样本索引
            "scores": scores,  # 相似度分数
        })
    
    return results

