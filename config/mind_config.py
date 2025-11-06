"""
MIND 框架配置
"""
# SSR (Similar Sample Retrieval) 配置
SSR_K = 10  # 相似样本数量
SSR_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # sentence-transformers 模型
SSR_TFIDF_MAX_FEATURES = 5000  # TF-IDF 最大特征数

# RID (Relevant Insight Derivation) 配置
RID_NUM_EXAMPLES = 3  # 用于规则生成的样本数量
RID_MAX_RULES = 5  # 最大规则数量

# IAI (Insight-Augmented Inference) 配置
IAI_MAX_THOUGHT_LENGTH = 1200  # 思考过程最大长度

