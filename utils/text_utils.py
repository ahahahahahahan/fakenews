"""
文本处理工具函数
"""


def extract_answer(response: str) -> str:
    """
    从响应中提取答案
    
    Args:
        response: API 响应文本
        
    Returns:
        提取的答案（"真实"、"虚假"或"未知"）
    """
    response_lower = response.lower()
    if "答案：" in response:
        answer = response.split("答案：")[-1].split('.')[0].strip().strip('[').strip(']')
    elif "answer:" in response_lower:
        answer = response_lower.split("answer:")[-1].split('.')[0].strip().strip('[').strip(']')
    else:
        # 尝试直接查找关键词
        if "虚假" in response or "false" in response_lower:
            answer = "虚假"
        elif "真实" in response or "true" in response_lower:
            answer = "真实"
        else:
            answer = "未知"
    return answer


def extract_thought(response: str) -> str:
    """
    从响应中提取思考过程
    
    Args:
        response: API 响应文本
        
    Returns:
        提取的思考过程
    """
    if "思考：" in response:
        return response.split("思考：")[-1]
    elif "thought:" in response.lower():
        return response.split("thought:")[-1]
    return response


def extract_rules_from_response(response: str) -> str:
    """
    从 RID 响应中提取规则
    
    Args:
        response: RID API 响应文本
        
    Returns:
        提取的规则文本
    """
    if "更新后的规则：" in response:
        return response.split("更新后的规则：", 1)[-1].strip()
    elif "更新后的规则" in response:
        # 尝试其他可能的格式
        parts = response.split("更新后的规则")
        if len(parts) > 1:
            return parts[-1].strip().strip("：").strip(":").strip()
    return "尚无规则。"

