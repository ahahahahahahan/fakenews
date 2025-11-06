"""
API 配置
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API 配置
API_KEY = os.getenv("YUNWU_API_KEY")
API_URL = "https://api.yunwu.ai/v1/chat/completions"
API_MODEL = "gemini-2.5-flash-lite-preview-06-17"
API_TIMEOUT = 30  # 超时时间（秒）

