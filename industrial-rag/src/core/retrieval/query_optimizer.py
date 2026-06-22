"""
查询优化模块

在检索前对用户查询进行优化，提高检索质量：
1. 查询扩展（Query Expansion）- 补充同义词、相关概念
2. 查询改写（Query Rewriting）- 纠正错误、补全语义
3. 多查询生成（Multi-Query）- 从不同角度生成多个查询
"""
from src.models.llm import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryOptimizer:
    """查询优化器。"""

    def __init__(self):
        self.llm = get_llm_client()

    async def expand_query(self, query: str) -> str:
        """
        查询扩展：补充同义词和相关概念。

        Args:
            query: 原始查询

        Returns:
            扩展后的查询（保留原查询 + 补充词）
        """
        prompt = f"""请为以下查询补充同义词和相关概念，帮助更全面地检索信息。

原始查询：{query}

要求：
1. 保留原查询的核心意图
2. 添加2-3个同义词或相关概念
3. 用空格分隔，直接输出扩展后的查询，不要解释

扩展查询："""

        try:
            expanded = await self.llm.generate(prompt=prompt, max_tokens=100)
            expanded = expanded.strip()
            logger.info(f"Query expanded: '{query}' → '{expanded}'")
            return expanded
        except Exception as e:
            logger.error(f"Query expansion failed: {e}, using original query")
            return query

    async def rewrite_query(self, query: str) -> str:
        """
        查询改写：纠正语法错误、补全不完整的语义。

        Args:
            query: 原始查询

        Returns:
            改写后的查询
        """
        prompt = f"""请改写以下查询，使其更清晰、准确，适合检索系统理解。

原始查询：{query}

要求：
1. 纠正语法错误
2. 补全不完整的语义
3. 保留用户意图
4. 直接输出改写后的查询，不要解释

改写查询："""

        try:
            rewritten = await self.llm.generate(prompt=prompt, max_tokens=100)
            rewritten = rewritten.strip()
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using original query")
            return query

    async def generate_multi_queries(self, query: str, num_queries: int = 3) -> list[str]:
        """
        多查询生成：从不同角度生成多个查询，提高召回率。

        Args:
            query: 原始查询
            num_queries: 生成查询数量

        Returns:
            多个查询（含原查询）
        """
        prompt = f"""请从不同角度改写以下查询，生成{num_queries}个不同但相关的查询，以提高检索全面性。

原始查询：{query}

要求：
1. 每个查询从不同角度或层次理解原问题
2. 保持查询的相关性和准确性
3. 每行一个查询
4. 不要编号，不要解释

查询列表："""

        try:
            response = await self.llm.generate(prompt=prompt, max_tokens=200)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            # 补充原查询（如果不在结果中）
            if query not in queries:
                queries.insert(0, query)
            queries = queries[: num_queries + 1]  # 限制数量
            logger.info(f"Generated {len(queries)} queries from: '{query}'")
            return queries
        except Exception as e:
            logger.error(f"Multi-query generation failed: {e}, using original query only")
            return [query]

    async def optimize_query(
        self,
        query: str,
        strategy: str = "rewrite",
    ) -> str | list[str]:
        """
        统一查询优化入口。

        Args:
            query: 原始查询
            strategy: 优化策略（"expand" / "rewrite" / "multi"）

        Returns:
            优化后的查询（rewrite/expand）或多个查询（multi）
        """
        if strategy == "expand":
            return await self.expand_query(query)
        elif strategy == "rewrite":
            return await self.rewrite_query(query)
        elif strategy == "multi":
            return await self.generate_multi_queries(query)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using original query")
            return query
