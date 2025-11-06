"""
新闻事实检查器主程序
"""
from models import NewsFactChecker

if __name__ == "__main__":
    # 使用 MIND 框架
    checker_mind = NewsFactChecker(use_mind=True)
    
    # 需要提供训练数据和测试数据
    result_mind = checker_mind.run(
        test_data_path="test_dataset.pkl",
        train_data_path="train_dataset.pkl",
        dataset_type="politifact"
    )
    
    # 或者使用简单模式（不需要训练数据）
    checker_simple = NewsFactChecker(use_mind=False)
    result_simple = checker_simple.run(
        test_data_path="test_dataset.pkl",
        dataset_type="politifact"
    )
