"""
API 调用工具函数
"""
import asyncio
import aiohttp
from typing import Optional
from config import API_KEY, API_URL, API_MODEL, API_TIMEOUT, TEMPERATURE, MAX_TOKENS, REQUEST_DELAY


async def fetch_api(
    session: aiohttp.ClientSession,
    prompt: str,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    timeout: int = API_TIMEOUT,
    request_delay: float = REQUEST_DELAY
) -> str:
    """
    异步调用 API
    
    Args:
        session: aiohttp 会话对象
        prompt: 提示文本
        temperature: 温度参数
        max_tokens: 最大 token 数
        timeout: 超时时间（秒）
        request_delay: 请求延迟（秒）
        
    Returns:
        API 响应文本
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        await asyncio.sleep(request_delay)
        async with session.post(API_URL, headers=headers, json=payload, timeout=timeout) as response:
            if response.status == 200:
                result = await response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
            return "API请求失败"
    except Exception as e:
        print(f"API调用失败：{e}")
        return f"API调用失败: {e}"

