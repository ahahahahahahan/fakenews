"""
新闻事实检查器模型
"""
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

# 导入配置
from config import (
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS,
    SSR_K,
    RID_NUM_EXAMPLES, RID_MAX_RULES,
    IAI_MAX_THOUGHT_LENGTH,
    FAKE_INDICATORS, REAL_INDICATORS,
    RID_PROMPT_TEMPLATE,
    IAI_DEBATER_PROMPT_TEMPLATE,
    IAI_JUDGE_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE
)

# 导入工具函数
from utils import (
    preprocess_data,
    extract_answer,
    extract_thought,
    extract_rules_from_response,
    calculate_metrics,
    EmbeddingComputer,
    compute_similarity_scores,
    extract_top_k_similar,
    fetch_api
)


class NewsFactChecker:
    def __init__(self, use_mind: bool = True):
        """
        初始化新闻事实检查器
        
        Args:
            use_mind: 是否使用 MIND 框架的三阶段逻辑（SSR + RID + IAI）
        """
        self.use_mind = use_mind
        
        # 从配置加载参数
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.timeout = API_TIMEOUT
        self.batch_size = BATCH_SIZE
        self.max_concurrency = MAX_CONCURRENCY
        self.batch_delay = BATCH_DELAY
        self.request_delay = REQUEST_DELAY
        
        # MIND 框架配置
        self.ssr_k = SSR_K
        self.rid_num_examples = RID_NUM_EXAMPLES
        self.max_rules = RID_MAX_RULES
        
        # 检测指标
        self.fake_indicators = FAKE_INDICATORS
        self.real_indicators = REAL_INDICATORS
        
        # 初始化文本嵌入模型（用于 SSR）
        if self.use_mind:
            self.embedding_computer = EmbeddingComputer()
        else:
            self.embedding_computer = None
        
        # 存储训练数据（用于 SSR）
        self.train_data = None
        self.train_embeddings = None
        self.train_texts = None

    def ssr_similar_sample_retrieval(self, test_texts: List[str], train_df: Any) -> List[Dict]:
        """
        SSR: 相似样本检索 (Similar Sample Retrieval)
        使用文本嵌入模型找到每个测试样本的 top-k 相似训练样本
        """
        print(f"\n--- SSR: 开始相似样本检索 ---")
        
        # 准备训练数据
        self.train_texts = train_df["text"].tolist()
        
        # 计算训练数据嵌入
        print("计算训练数据嵌入...")
        self.train_embeddings = self.embedding_computer.compute_embeddings(self.train_texts)
        
        # 计算测试数据嵌入
        print("计算测试数据嵌入...")
        test_embeddings = self.embedding_computer.compute_embeddings(test_texts)
        
        # 计算相似度
        print("计算相似度分数...")
        similarity_scores = compute_similarity_scores(test_embeddings, self.train_embeddings)
        
        # 提取 top-k 相似样本
        print(f"提取 top-{self.ssr_k} 相似样本...")
        results = extract_top_k_similar(similarity_scores, self.ssr_k)
        
        print(f"✅ SSR 完成: {len(results)} 个测试样本")
        return results

    def _build_rid_prompt(self, text: str, existing_rules: str) -> str:
        """构建 RID 阶段的 prompt"""
        return RID_PROMPT_TEMPLATE.format(text=text, existing_rules=existing_rules).strip()

    async def _fetch_api(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """异步调用 API"""
        return await fetch_api(
            session=session,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            request_delay=self.request_delay
        )

    async def rid_relevant_insight_derivation(self, ssr_results: List[Dict], train_df: pd.DataFrame) -> List[Dict]:
        """
        RID: 相关洞察推导 (Relevant Insight Derivation)
        基于 SSR 找到的相似样本，迭代生成规则（正向和反向）
        """
        print(f"\n--- RID: 开始相关洞察推导 ---")
        
        train_texts = train_df["text"].tolist()
        rid_results = []
        
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def generate_rules_async(indices: List[int], direction: str) -> str:
            """异步生成规则"""
            rules = "尚无规则。"
            
            items_to_process = []
            for idx in indices[:self.rid_num_examples]:
                if 0 <= idx < len(train_texts):
                    text = train_texts[idx]
                    prompt = self._build_rid_prompt(text, rules)
                    items_to_process.append({
                        'prompt': prompt,
                        'index': idx
                    })
            
            if not items_to_process:
                return rules
            
            async def bounded_fetch(item):
                async with semaphore:
                    return await self._fetch_api(session, item['prompt'])
            
            tasks = [bounded_fetch(item) for item in items_to_process]
            responses = await asyncio.gather(*tasks)
            
            # 处理响应并更新规则
            for response in responses:
                extracted_rules = extract_rules_from_response(response)
                if extracted_rules != "尚无规则。":
                    rules = extracted_rules
            
            return rules
        
        async with aiohttp.ClientSession() as session:
            for ssr_item in ssr_results:
                test_idx = ssr_item['index']
                sample_indices = ssr_item['samples'][:self.rid_num_examples]
                reversed_indices = sample_indices[::-1]
                
                # 并发生成正向和反向规则
                forward_rules, backward_rules = await asyncio.gather(
                    generate_rules_async(sample_indices, "forward"),
                    generate_rules_async(reversed_indices, "backward")
                )
                
                rid_results.append({
                    'index': test_idx,
                    'forward': forward_rules,
                    'backward': backward_rules
                })
        
        print(f"✅ RID 完成: {len(rid_results)} 个测试样本")
        return rid_results

    def _build_iai_debater_prompt(self, text: str, rules: str) -> str:
        """构建 IAI 辩论者的 prompt"""
        return IAI_DEBATER_PROMPT_TEMPLATE.format(text=text, rules=rules).strip()

    def _build_iai_judge_prompt(self, text: str, predict_1: str, thought_1: str, 
                                predict_2: str, thought_2: str) -> str:
        """构建 IAI 法官的 prompt"""
        # 截断思考过程长度
        thought_1_truncated = thought_1[:IAI_MAX_THOUGHT_LENGTH]
        thought_2_truncated = thought_2[:IAI_MAX_THOUGHT_LENGTH]
        return IAI_JUDGE_PROMPT_TEMPLATE.format(
            text=text,
            predict_1=predict_1,
            thought_1=thought_1_truncated,
            predict_2=predict_2,
            thought_2=thought_2_truncated
        ).strip()

    def _extract_answer(self, response: str) -> str:
        """从响应中提取答案"""
        return extract_answer(response)

    def _extract_thought(self, response: str) -> str:
        """从响应中提取思考过程"""
        return extract_thought(response)

    async def iai_insight_augmented_inference(self, test_texts: List[str], rid_results: List[Dict]) -> List[Dict]:
        """
        IAI: 洞察增强推理 (Insight-Augmented Inference)
        使用 RID 生成的规则进行多智能体推理（两个辩论者+法官）
        """
        print(f"\n--- IAI: 开始洞察增强推理 ---")
        
        iai_results = []
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def process_item(test_idx: int, text: str, rid_item: Dict):
            """处理单个测试项"""
            forward_rules = rid_item.get('forward', '尚无规则。')
            backward_rules = rid_item.get('backward', '尚无规则。')
            
            async def bounded_fetch(prompt):
                async with semaphore:
                    return await self._fetch_api(session, prompt)
            
            # 并发调用两个辩论者
            debater1_prompt = self._build_iai_debater_prompt(text, forward_rules)
            debater2_prompt = self._build_iai_debater_prompt(text, backward_rules)
            
            debater1_response, debater2_response = await asyncio.gather(
                bounded_fetch(debater1_prompt),
                bounded_fetch(debater2_prompt)
            )
            
            predict_1 = self._extract_answer(debater1_response)
            thought_1 = self._extract_thought(debater1_response)
            predict_2 = self._extract_answer(debater2_response)
            thought_2 = self._extract_thought(debater2_response)
            
            # 如果两个辩论者意见一致，直接使用结果
            if predict_1 == predict_2:
                final_predict = predict_1
                judge_response = None
            else:
                # 调用法官
                judge_prompt = self._build_iai_judge_prompt(text, predict_1, thought_1, predict_2, thought_2)
                judge_response = await bounded_fetch(judge_prompt)
                final_predict = self._extract_answer(judge_response)
            
            return {
                'index': test_idx,
                'text': text,
                'predict': final_predict,
                'debater1_predict': predict_1,
                'debater2_predict': predict_2,
                'debater1_thought': thought_1,
                'debater2_thought': thought_2,
                'judge_response': judge_response
            }
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, (text, rid_item) in enumerate(zip(test_texts, rid_results)):
                if rid_item['index'] == i:
                    tasks.append(process_item(i, text, rid_item))
            
            iai_results = await asyncio.gather(*tasks)
        
        print(f"✅ IAI 完成: {len(iai_results)} 个测试样本")
        return iai_results

    def calculate_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
        """计算评估指标"""
        return calculate_metrics(true_labels, pred_labels)

    async def run_with_mind(self, test_data_path: str, train_data_path: str, 
                           dataset_type: str = "politifact") -> Dict[str, Any]:
        """
        使用 MIND 框架的完整运行流程
        SSR → RID → IAI
        """
        start_time = time.time()
        
        # 根据数据集类型调整配置
        if dataset_type in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_type]
            self.batch_size = config["batch_size"]
            self.max_concurrency = config["max_concurrency"]
        
        # 1. 数据预处理
        print("--- 步骤 1: 数据预处理 ---")
        test_df = preprocess_data(test_data_path, dataset_type)
        train_df = preprocess_data(train_data_path, dataset_type)
        
        test_texts = test_df["text"].tolist()
        true_labels = test_df["label"].tolist() if "label" in test_df.columns else []
        
        # 2. SSR: 相似样本检索
        ssr_results = self.ssr_similar_sample_retrieval(test_texts, train_df)
        
        # 3. RID: 相关洞察推导
        rid_results = await self.rid_relevant_insight_derivation(ssr_results, train_df)
        
        # 4. IAI: 洞察增强推理
        iai_results = await self.iai_insight_augmented_inference(test_texts, rid_results)
        
        # 5. 提取预测结果
        pred_labels = [result['predict'] for result in iai_results]
        
        # 6. 计算指标
        metrics = {}
        if true_labels:
            metrics = self.calculate_metrics(true_labels, pred_labels)
        
        total_time = time.time() - start_time
        
        result = {
            "dataset_type": dataset_type,
            "total_samples": len(test_texts),
            "total_time": round(total_time, 2),
            "process_speed": round(len(test_texts) / total_time, 2) if total_time > 0 else 0,
            "metrics": metrics,
            "predictions": pred_labels
        }
        
        print(f"\n✅ MIND 框架处理完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"指标: {metrics}")
        
        return result

    async def run_simple(self, data_path: str, dataset_type: str = "politifact") -> Dict[str, Any]:
        """
        简单模式：直接使用 API 进行检测（原始逻辑）
        """
        start_time = time.time()
        
        # 根据数据集类型调整配置
        if dataset_type in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_type]
            self.batch_size = config["batch_size"]
            self.max_concurrency = config["max_concurrency"]
        
        # 数据预处理
        df = preprocess_data(data_path, dataset_type)
        texts = df["text"].tolist()
        true_labels = df["label"].tolist() if "label" in df.columns else []
        
        # 构建简单的 prompt
        fake_indicators_str = "、".join(self.fake_indicators)
        real_indicators_str = "、".join(self.real_indicators)
        
        def build_simple_prompt(text: str) -> str:
            return SIMPLE_PROMPT_TEMPLATE.format(
                text=text,
                fake_indicators=fake_indicators_str,
                real_indicators=real_indicators_str
            )
        
        # 异步处理
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def fetch_simple(session, text):
            async with semaphore:
                await asyncio.sleep(self.request_delay)
                prompt = build_simple_prompt(text)
                return await self._fetch_api(session, prompt)
        
        async with aiohttp.ClientSession() as session:
            batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
            pred_labels = []
            
            for batch in batches:
                tasks = [fetch_simple(session, text) for text in batch]
                batch_results = await asyncio.gather(*tasks)
                pred_labels.extend(batch_results)
                await asyncio.sleep(self.batch_delay)
        
        # 计算指标
        metrics = {}
        if true_labels:
            metrics = self.calculate_metrics(true_labels, pred_labels)
        
        total_time = time.time() - start_time
        
        result = {
            "dataset_type": dataset_type,
            "total_samples": len(texts),
            "total_time": round(total_time, 2),
            "process_speed": round(len(texts) / total_time, 2) if total_time > 0 else 0,
            "metrics": metrics
        }
        
        print(f"✅ 简单模式处理完成！")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"指标: {metrics}")
        
        return result

    def run(self, test_data_path: str, train_data_path: str = None, 
            dataset_type: str = "politifact") -> Dict[str, Any]:
        """
        主运行函数
        
        Args:
            test_data_path: 测试数据路径
            train_data_path: 训练数据路径（MIND 模式必需）
            dataset_type: 数据集类型
        """
        if self.use_mind and train_data_path:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.run_with_mind(test_data_path, train_data_path, dataset_type)
            )
        else:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.run_simple(test_data_path, dataset_type)
            )

