"""
新闻事实检查器模型
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import time
import copy
from typing import List, Dict, Tuple, Any
from numpy.linalg import norm
from sklearn.metrics import f1_score

# 导入配置
from config import (
    API_KEY, API_URL, API_MODEL, API_TIMEOUT,
    TEMPERATURE, MAX_TOKENS,
    BATCH_SIZE, MAX_CONCURRENCY, BATCH_DELAY, REQUEST_DELAY,
    DATASET_CONFIGS,
    SSR_K, SSR_EMBEDDING_MODEL, SSR_TFIDF_MAX_FEATURES,
    RID_NUM_EXAMPLES, RID_MAX_RULES,
    IAI_MAX_THOUGHT_LENGTH,
    FAKE_INDICATORS, REAL_INDICATORS,
    RID_PROMPT_TEMPLATE,
    IAI_DEBATER_PROMPT_TEMPLATE,
    IAI_JUDGE_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE
)

# 尝试导入 sentence-transformers，如果失败则使用简单的 TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except ImportError:
    print("警告: sentence-transformers 未安装，将使用简单的文本相似度方法")
    USE_SBERT = False
    from sklearn.feature_extraction.text import TfidfVectorizer


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
            if USE_SBERT:
                print("加载 sentence-transformers 模型用于 SSR...")
                self.embedding_model = SentenceTransformer(SSR_EMBEDDING_MODEL)
            else:
                print("使用 TF-IDF 进行文本相似度计算...")
                self.embedding_model = None
                self.tfidf_vectorizer = TfidfVectorizer(max_features=SSR_TFIDF_MAX_FEATURES)
        
        # 存储训练数据（用于 SSR）
        self.train_data = None
        self.train_embeddings = None
        self.train_texts = None

    def preprocess_data(self, data_path: str, dataset_type: str = "politifact") -> pd.DataFrame:
        """数据预处理"""
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

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """计算文本嵌入向量"""
        if USE_SBERT and self.embedding_model:
            return self.embedding_model.encode(texts, show_progress_bar=False)
        else:
            # 使用 TF-IDF
            vectors = self.tfidf_vectorizer.fit_transform(texts)
            return vectors.toarray()

    def ssr_similar_sample_retrieval(self, test_texts: List[str], train_df: pd.DataFrame) -> List[Dict]:
        """
        SSR: 相似样本检索 (Similar Sample Retrieval)
        使用文本嵌入模型找到每个测试样本的 top-k 相似训练样本
        """
        print(f"\n--- SSR: 开始相似样本检索 ---")
        
        # 准备训练数据
        self.train_texts = train_df["text"].tolist()
        train_labels = train_df["label"].tolist() if "label" in train_df.columns else [None] * len(self.train_texts)
        
        # 计算训练数据嵌入
        print("计算训练数据嵌入...")
        self.train_embeddings = self._compute_embeddings(self.train_texts)
        
        # 计算测试数据嵌入
        print("计算测试数据嵌入...")
        test_embeddings = self._compute_embeddings(test_texts)
        
        # 计算相似度
        print("计算相似度分数...")
        similarity_scores = np.zeros((len(test_embeddings), len(self.train_embeddings)))
        
        dot_products = np.dot(test_embeddings, self.train_embeddings.T)
        norms_test = norm(test_embeddings, axis=1, keepdims=True)
        norms_train = norm(self.train_embeddings, axis=1, keepdims=True).T
        
        norms_test[norms_test == 0] = 1
        norms_train[norms_train == 0] = 1
        similarity_scores = dot_products / (norms_test * norms_train)
        similarity_scores = np.clip(similarity_scores, -1.0, 1.0)
        similarity_scores[similarity_scores >= 1.0] = 0.0
        
        # 提取 top-k 相似样本
        similarity_scores_copy = copy.deepcopy(similarity_scores)
        results = []
        
        print(f"提取 top-{self.ssr_k} 相似样本...")
        for i in range(len(test_embeddings)):
            samples = []
            scores = []
            current_scores = similarity_scores_copy[i]
            
            for _ in range(self.ssr_k):
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
        
        print(f"✅ SSR 完成: {len(results)} 个测试样本")
        return results

    def _build_rid_prompt(self, text: str, existing_rules: str) -> str:
        """构建 RID 阶段的 prompt"""
        return RID_PROMPT_TEMPLATE.format(text=text, existing_rules=existing_rules).strip()

    async def _fetch_api(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """异步调用 API"""
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": API_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            await asyncio.sleep(self.request_delay)
            async with session.post(API_URL, headers=headers, json=payload, timeout=self.timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content'].strip()
                return "API请求失败"
        except Exception as e:
            print(f"API调用失败：{e}")
            return f"API调用失败: {e}"

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
                if "更新后的规则：" in response:
                    rules = response.split("更新后的规则：", 1)[-1].strip()
                elif "更新后的规则" in response:
                    # 尝试其他可能的格式
                    parts = response.split("更新后的规则")
                    if len(parts) > 1:
                        rules = parts[-1].strip().strip("：").strip(":").strip()
            
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

    def _extract_thought(self, response: str) -> str:
        """从响应中提取思考过程"""
        if "思考：" in response:
            return response.split("思考：")[-1]
        elif "thought:" in response.lower():
            return response.split("thought:")[-1]
        return response

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
        # 统一标签格式
        true_labels_processed = []
        pred_labels_processed = []
        
        for t, p in zip(true_labels, pred_labels):
            # 处理真实标签
            t_str = str(t).lower()
            if t_str in ["虚假", "false", "1", "fake"]:
                true_labels_processed.append(1)
            elif t_str in ["真实", "true", "0", "real"]:
                true_labels_processed.append(0)
            else:
                continue
            
            # 处理预测标签
            p_str = str(p).lower()
            if p_str in ["虚假", "false", "1", "fake"]:
                pred_labels_processed.append(1)
            elif p_str in ["真实", "true", "0", "real"]:
                pred_labels_processed.append(0)
            else:
                # 如果无法识别，假设为错误预测
                pred_labels_processed.append(1 - true_labels_processed[-1])
        
        if len(true_labels_processed) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
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
        test_df = self.preprocess_data(test_data_path, dataset_type)
        train_df = self.preprocess_data(train_data_path, dataset_type)
        
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
        df = self.preprocess_data(data_path, dataset_type)
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

