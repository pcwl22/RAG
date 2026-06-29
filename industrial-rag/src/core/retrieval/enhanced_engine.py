"""Enhanced retrieval engine with query understanding and multi-query retrieval."""
import asyncio
from typing import Any

from src.core.config import get_settings
from src.core.retrieval.engine import RetrievalEngine
from src.core.retrieval.query_understanding import QueryUnderstanding
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedRetrievalEngine:
    """Retrieval engine with query understanding (coreference resolution and decomposition)."""

    def __init__(self):
        self.retrieval_engine = RetrievalEngine()
        self.query_understanding = QueryUnderstanding()
        self.config = get_settings().get("rag", {}).get("retrieval", {})

    async def retrieve_with_understanding(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        enable_rerank: bool | None = None,
        partition: str | None = None,
        enable_query_understanding: bool = True,
    ) -> dict[str, Any]:
        """Retrieve with query understanding (coreference + decomposition).

        Returns:
            {
                "query_understanding": {...},  # Query understanding results
                "results": [...],              # Final merged results
                "subquery_results": [...]      # Results per subquery (if decomposed)
            }
        """
        # Step 1: Query understanding
        understanding_result = {"original_query": query, "resolved_query": query, "subqueries": [query]}

        if enable_query_understanding:
            qu_config = self.config.get("query_understanding", {})
            enable_coreference = qu_config.get("enable_coreference", True)
            enable_decomposition = qu_config.get("enable_decomposition", True)
            enable_rewrite = qu_config.get("enable_rewrite", True)

            understanding_result = await self.query_understanding.understand_query(
                query=query,
                chat_history=chat_history,
                enable_coreference=enable_coreference,
                enable_decomposition=enable_decomposition,
                enable_rewrite=enable_rewrite,
            )

            logger.info(
                f"Query understanding: {len(understanding_result.get('retrieval_queries', []))} retrieval queries, "
                f"decomposed={understanding_result.get('is_decomposed', False)}"
            )

        subqueries = understanding_result.get("retrieval_queries") or understanding_result["subqueries"]

        # Step 2: Retrieve for each subquery
        if len(subqueries) == 1:
            # Single query - standard retrieval
            results = await self.retrieval_engine.retrieve(
                query=subqueries[0],
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                enable_rerank=enable_rerank,
                partition=partition,
            )
            return {
                "query_understanding": understanding_result,
                "results": results,
                "subquery_results": [{"subquery": subqueries[0], "results": results}],
            }

        # Multiple subqueries - parallel retrieval and merging
        logger.info(f"Retrieving for {len(subqueries)} subqueries in parallel")

        # Retrieve for all subqueries in parallel
        tasks = [
            self.retrieval_engine.retrieve(
                query=subquery,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                enable_rerank=enable_rerank,
                partition=partition,
            )
            for subquery in subqueries
        ]

        all_results = await asyncio.gather(*tasks)

        # Merge results from all subqueries
        merged_results = self._merge_subquery_results(all_results, top_k or self.config.get("top_k", 5))

        subquery_results = [
            {"subquery": sq, "results": res}
            for sq, res in zip(subqueries, all_results)
        ]

        return {
            "query_understanding": understanding_result,
            "results": merged_results,
            "subquery_results": subquery_results,
        }

    def _merge_subquery_results(self, all_results: list[list[dict]], top_k: int) -> list[dict]:
        """Merge results from multiple subqueries using score-based ranking.

        Deduplicates by chunk ID and re-ranks by max score across subqueries.
        """
        # Collect all unique chunks with their best scores
        chunk_scores: dict[str, dict] = {}

        for results in all_results:
            for doc in results:
                chunk_id = doc.get("id")
                if not chunk_id:
                    continue

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = {
                        "doc": doc,
                        "max_score": doc.get("score", 0.0),
                        "appearances": 1,
                    }
                else:
                    # Update max score if this is higher
                    current_score = doc.get("score", 0.0)
                    if current_score > chunk_scores[chunk_id]["max_score"]:
                        chunk_scores[chunk_id]["max_score"] = current_score
                        chunk_scores[chunk_id]["doc"] = doc
                    chunk_scores[chunk_id]["appearances"] += 1

        # Sort by max_score (and appearances as tiebreaker)
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: (x["max_score"], x["appearances"]),
            reverse=True,
        )

        # Take top_k and update their scores
        merged: list[dict] = []
        for item in sorted_chunks[:top_k]:
            doc = item["doc"].copy()
            doc["score"] = item["max_score"]
            doc["subquery_appearances"] = item["appearances"]
            merged.append(doc)

        logger.info(f"Merged {len(chunk_scores)} unique chunks into {len(merged)} results")
        return merged
