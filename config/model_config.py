"""
模型相关配置
"""
# LLM 模型配置
TEMPERATURE = 0.1
MAX_TOKENS = 500

# 异步处理配置
BATCH_SIZE = 30  # 默认批次大小
MAX_CONCURRENCY = 15  # 默认最大并发数
BATCH_DELAY = 2  # 批次间延迟（秒）
REQUEST_DELAY = 1  # 请求间延迟（秒）

# 数据集特定配置
DATASET_CONFIGS = {
    "politifact": {
        "batch_size": 30,
        "max_concurrency": 15
    },
    "weibo21": {
        "batch_size": 20,
        "max_concurrency": 8
    },
    "gossip": {
        "batch_size": 20,
        "max_concurrency": 8
    }
}

