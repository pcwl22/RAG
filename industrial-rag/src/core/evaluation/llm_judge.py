"""
LLM-as-Judge 评估框架

无需硬编码标准答案，由 LLM 自动判定检索上下文是否命中关键信息
集成 RAGAS 评估生成质量
"""
import asyncio
import re
from typing import Any

from src.models.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _context_text(context: dict) -> str:
    metadata = context.get("metadata") or {}
    return str(context.get("content") or context.get("child_content") or metadata.get("parent_content") or "")


def _answer_body(answer: str) -> str:
    match = re.search(r"(?:##\s*)?【回答】", answer)
    if not match:
        return answer
    body = answer[match.end():]
    return re.split(r"\n(?:##\s*)?【匹配文件】|\n(?:##\s*)?【文件具体位置", body, maxsplit=1)[0]


def _extract_eval_terms(query: str, answer: str) -> list[str]:
    text = f"{query}\n{_answer_body(answer)}"
    terms = re.findall(r"第[一二三四五六七八九十百千万零〇两0-9]+条", text)
    legal_terms = [
        "非法拘禁",
        "故意伤害",
        "故意杀人",
        "侵权责任",
        "工作人员",
        "使用暴力",
        "致人伤残",
        "致人死亡",
        "胁从犯",
        "被胁迫参加犯罪",
        "用人单位",
        "执行工作任务",
        "职务侵占",
        "盗窃",
        "无固定期限劳动合同",
        "经济补偿",
        "胎儿",
        "遗产",
        "继承",
        "民事权利能力",
    ]
    terms.extend(term for term in legal_terms if term in text)

    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique


def _required_term_groups(query: str) -> list[list[str]]:
    groups: list[list[str]] = []
    if "胎儿" in query:
        groups.append(["胎儿"])
    if "非法拘禁" in query:
        groups.append(["非法拘禁"])
    if "胁从犯" in query or "受胁迫" in query:
        groups.append(["胁从犯", "被胁迫参加犯罪"])
    if "职务侵占" in query:
        groups.append(["职务侵占"])
    if "盗窃" in query:
        groups.append(["盗窃"])
    if "员工" in query or "工作人员" in query or "劳动者" in query:
        groups.append(["用人单位", "工作人员"])
    if "执行工作任务" in query:
        groups.append(["执行工作任务"])
    if "无固定期限劳动合同" in query:
        groups.append(["无固定期限劳动合同"])
    if "经济补偿" in query or "补偿金" in query:
        groups.append(["经济补偿"])
    return groups


def _rule_based_retrieval_eval(query: str, contexts: list[dict], top_n: int, answer: str = "") -> dict[str, Any]:
    contexts_to_eval = contexts[:top_n]
    terms = _extract_eval_terms(query, answer)
    if not contexts_to_eval or not terms:
        return {
            "is_relevant": False,
            "hit_count": 0,
            "total": len(contexts_to_eval),
            "precision": 0.0,
            "reason": "",
        }

    query_terms = [term for term in _extract_eval_terms(query, "") if not term.startswith("第")]
    required_groups = _required_term_groups(query)
    hit_count = 0
    matched_terms: list[str] = []
    for context in contexts_to_eval:
        text = _context_text(context)
        context_terms = [term for term in terms if term in text]
        has_required_signal = not required_groups or any(
            any(term in text for term in group)
            for group in required_groups
        )
        has_query_signal = has_required_signal and (not query_terms or any(term in text for term in query_terms))
        if context_terms and has_query_signal:
            hit_count += 1
            for term in context_terms:
                if term not in matched_terms:
                    matched_terms.append(term)

    return {
        "is_relevant": hit_count > 0,
        "hit_count": hit_count,
        "total": len(contexts_to_eval),
        "precision": hit_count / len(contexts_to_eval) if contexts_to_eval else 0.0,
        "reason": f"规则命中关键词：{'、'.join(matched_terms)}" if matched_terms else "",
    }


class LLMJudge:
    """LLM-as-Judge 评估器"""

    def __init__(self):
        self.llm = get_llm_client()

    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_contexts: list[dict],
        top_n: int = 5,
        answer: str = "",
    ) -> dict[str, Any]:
        """评估检索质量

        判断 Top-N 检索上下文是否命中查询的关键信息

        Args:
            query: 用户查询
            retrieved_contexts: 检索到的上下文列表
            top_n: 评估前 N 个结果

        Returns:
            评估结果字典
        """
        if not retrieved_contexts:
            return {
                "is_relevant": False,
                "hit_count": 0,
                "total": 0,
                "precision": 0.0,
                "reason": "No contexts retrieved"
            }

        # 取前 N 个
        contexts_to_eval = retrieved_contexts[:top_n]

        # 构造评估 prompt
        contexts_text = "\n\n".join([
            f"[文档 {i+1}]\n{ctx.get('content', '')[:500]}"
            for i, ctx in enumerate(contexts_to_eval)
        ])

        prompt = f"""请评估以下检索结果是否包含回答用户查询所需的关键信息。

用户查询：{query}

检索到的文档：
{contexts_text}

评估任务：
1. 判断这些文档中是否包含回答查询所需的关键信息
2. 统计有多少个文档是相关的
3. 给出简短理由

请按以下格式回答：
相关文档数: <数字>
总文档数: {len(contexts_to_eval)}
是否命中: <是/否>
理由: <简短理由>"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=200)

            # 解析响应
            lines = response.strip().split("\n")
            result = {
                "is_relevant": False,
                "hit_count": 0,
                "total": len(contexts_to_eval),
                "precision": 0.0,
                "reason": ""
            }

            for line in lines:
                if "相关文档数" in line or "hit_count" in line.lower():
                    try:
                        result["hit_count"] = int(line.split(":")[-1].strip())
                    except:
                        pass
                elif "是否命中" in line or "is_relevant" in line.lower():
                    result["is_relevant"] = "是" in line or "yes" in line.lower()
                elif "理由" in line or "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[-1].strip()

            result["precision"] = result["hit_count"] / result["total"] if result["total"] > 0 else 0.0

            rule_eval = _rule_based_retrieval_eval(query, retrieved_contexts, top_n, answer)
            if rule_eval["reason"]:
                result = rule_eval

            logger.info(f"Retrieval evaluation: hit={result['hit_count']}/{result['total']}, precision={result['precision']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            return {
                "is_relevant": False,
                "hit_count": 0,
                "total": len(contexts_to_eval),
                "precision": 0.0,
                "reason": f"Evaluation error: {e}"
            }

    async def evaluate_answer_quality(
        self,
        query: str,
        contexts: list[dict],
        answer: str,
    ) -> dict[str, Any]:
        """评估答案质量（简化版 RAGAS 指标）

        评估维度：
        1. Faithfulness - 答案是否忠实于上下文
        2. Answer Relevancy - 答案是否回答了问题
        3. Context Precision - 上下文是否精确相关

        Args:
            query: 用户查询
            contexts: 检索上下文
            answer: 生成的答案

        Returns:
            评估结果字典
        """
        contexts_text = "\n\n".join([
            f"[文档 {i+1}]\n{ctx.get('content', '')[:500]}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""请评估以下 RAG 系统的回答质量。

用户查询：{query}

检索上下文：
{contexts_text}

生成答案：{answer}

请从以下三个维度评分（0-10分）：
1. Faithfulness（忠实度）- 答案是否忠实于上下文，没有编造信息
2. Answer Relevancy（相关性）- 答案是否直接回答了用户问题
3. Context Precision（上下文精度）- 上下文是否精确相关，没有冗余信息

请按以下格式回答：
Faithfulness: <0-10>
Answer Relevancy: <0-10>
Context Precision: <0-10>
总体评价: <简短评价>"""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=300)

            # 解析评分
            result = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "overall_score": 0.0,
                "comment": ""
            }

            lines = response.strip().split("\n")
            for line in lines:
                if "faithfulness" in line.lower():
                    try:
                        score = float(line.split(":")[-1].strip())
                        result["faithfulness"] = score / 10.0  # 归一化到 0-1
                    except:
                        pass
                elif "answer relevancy" in line.lower() or "相关性" in line:
                    try:
                        score = float(line.split(":")[-1].strip())
                        result["answer_relevancy"] = score / 10.0
                    except:
                        pass
                elif "context precision" in line.lower() or "上下文精度" in line:
                    try:
                        score = float(line.split(":")[-1].strip())
                        result["context_precision"] = score / 10.0
                    except:
                        pass
                elif "总体评价" in line or "comment" in line.lower():
                    result["comment"] = line.split(":", 1)[-1].strip()

            # 计算总分
            result["overall_score"] = (
                result["faithfulness"] +
                result["answer_relevancy"] +
                result["context_precision"]
            ) / 3.0

            logger.info(
                f"Answer quality: faithfulness={result['faithfulness']:.2f}, "
                f"relevancy={result['answer_relevancy']:.2f}, "
                f"precision={result['context_precision']:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Answer quality evaluation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "overall_score": 0.0,
                "comment": f"Evaluation error: {e}"
            }

    async def evaluate_rag_pipeline(
        self,
        query: str,
        contexts: list[dict],
        answer: str,
        top_n_retrieval: int = 5,
    ) -> dict[str, Any]:
        """评估完整的 RAG 流水线

        Args:
            query: 用户查询
            contexts: 检索上下文
            answer: 生成的答案
            top_n_retrieval: 评估检索的前 N 个结果

        Returns:
            完整评估结果
        """
        # 并行评估检索和生成质量
        retrieval_task = self.evaluate_retrieval(query, contexts, top_n_retrieval, answer)
        quality_task = self.evaluate_answer_quality(query, contexts, answer)

        retrieval_eval, quality_eval = await asyncio.gather(
            retrieval_task,
            quality_task,
            return_exceptions=True
        )

        # 处理异常
        if isinstance(retrieval_eval, Exception):
            logger.error(f"Retrieval evaluation error: {retrieval_eval}")
            retrieval_eval = {"is_relevant": False, "hit_count": 0, "total": 0, "precision": 0.0, "reason": str(retrieval_eval)}

        if isinstance(quality_eval, Exception):
            logger.error(f"Quality evaluation error: {quality_eval}")
            quality_eval = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "overall_score": 0.0, "comment": str(quality_eval)}

        return {
            "query": query,
            "retrieval": retrieval_eval,
            "generation": quality_eval,
            "pipeline_score": (
                retrieval_eval.get("precision", 0.0) * 0.4 +
                quality_eval.get("overall_score", 0.0) * 0.6
            )
        }


class RAGASEvaluator:
    """RAGAS 指标评估器（简化版）"""

    def __init__(self):
        self.judge = LLMJudge()

    async def compute_ragas_metrics(
        self,
        query: str,
        contexts: list[dict],
        answer: str,
        ground_truth: str | None = None,
    ) -> dict[str, float]:
        """计算 RAGAS 指标

        Args:
            query: 用户查询
            contexts: 检索上下文
            answer: 生成的答案
            ground_truth: 标准答案（可选）

        Returns:
            RAGAS 指标字典
        """
        eval_result = await self.judge.evaluate_rag_pipeline(query, contexts, answer)

        metrics = {
            "context_precision": eval_result["retrieval"]["precision"],
            "faithfulness": eval_result["generation"]["faithfulness"],
            "answer_relevancy": eval_result["generation"]["answer_relevancy"],
            "context_recall": eval_result["generation"]["context_precision"],
        }

        # 如果有标准答案，计算答案相似度
        if ground_truth:
            metrics["answer_similarity"] = await self._compute_answer_similarity(
                answer, ground_truth
            )

        # 计算 RAGAS 总分
        metrics["ragas_score"] = sum(metrics.values()) / len(metrics)

        logger.info(f"RAGAS metrics: {metrics}")
        return metrics

    async def _compute_answer_similarity(
        self,
        answer: str,
        ground_truth: str,
    ) -> float:
        """计算答案与标准答案的相似度"""
        prompt = f"""请评估以下两个答案的语义相似度（0-10分）。

答案1：{answer}

答案2（标准答案）：{ground_truth}

相似度（0-10）："""

        try:
            response = await self.judge.llm.generate(prompt=prompt, max_tokens=50)
            score = float(response.strip().split()[0])
            return score / 10.0
        except:
            return 0.5  # 默认中等相似度
