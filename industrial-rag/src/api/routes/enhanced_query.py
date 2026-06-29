"""Enhanced query API integrating query understanding and evaluation."""
import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.core.evaluation.llm_judge import LLMJudge, RAGASEvaluator
from src.core.generation.generator import Generator
from src.core.retrieval.hybrid_engine import HybridRetrievalEngine
from src.core.retrieval.query_understanding import QueryUnderstanding
from src.models.llm import get_llm_client
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _stream_text_chunks(text: str, chunk_size: int = 24):
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


class EnhancedQueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    chat_history: list[dict[str, str]] | None = Field(None, description="Conversation history")
    enable_coreference: bool = True
    enable_decomposition: bool = True
    enable_rewrite: bool = True
    enable_evaluation: bool = False
    stream: bool = False
    top_k: int = Field(5, ge=1, le=20)
    partition: str | None = None


class EnhancedQueryResponse(BaseModel):
    query: str
    understanding: dict[str, Any]
    results: list[dict[str, Any]]
    answer: str
    sub_answers: list[dict[str, Any]] = []
    evaluation: dict[str, Any] | None = None


def _result_score(doc: dict[str, Any]) -> float:
    return float(doc.get("rrf_score", doc.get("score", 0.0)) or 0.0)


def _merge_results(
    result_groups: list[list[dict]],
    top_k: int,
    queries: list[str] | None = None,
) -> list[dict]:
    """Merge multi-query retrieval results by chunk identity and best score."""
    chunks: dict[str, dict[str, Any]] = {}

    for group_index, group in enumerate(result_groups):
        query = queries[group_index] if queries and group_index < len(queries) else None
        for rank, doc in enumerate(group):
            doc_id = str(doc.get("id"))
            if not doc_id:
                continue

            score = _result_score(doc)
            if doc_id not in chunks:
                chunks[doc_id] = {
                    "doc": doc,
                    "best_score": score,
                    "best_rank": rank,
                    "appearances": 1,
                    "matched_queries": [query] if query else [],
                }
                continue

            item = chunks[doc_id]
            item["appearances"] += 1
            if query and query not in item["matched_queries"]:
                item["matched_queries"].append(query)
            if (score, -rank) > (item["best_score"], -item["best_rank"]):
                item["doc"] = doc
                item["best_score"] = score
                item["best_rank"] = rank

    ranked = sorted(
        chunks.values(),
        key=lambda item: (item["best_score"], item["appearances"], -item["best_rank"]),
        reverse=True,
    )

    merged: list[dict] = []
    for item in ranked[:top_k]:
        doc = item["doc"].copy()
        doc["score"] = item["best_score"]
        doc["multi_query_appearances"] = item["appearances"]
        doc["matched_queries"] = item["matched_queries"]
        merged.append(doc)

    return merged


async def _aggregate_sub_answers(query: str, sub_answers: list[dict[str, Any]]) -> str:
    llm = get_llm_client()
    content = "\n\n".join(
        f"子问题 {index}: {item['query']}\n子答案: {item['answer']}"
        for index, item in enumerate(sub_answers, 1)
    )
    prompt = f"""请把以下多个子问题答案聚合成一个最终回答。

原始问题：
{query}

子问题答案：
{content}

要求：
1. 保留对子问题的关键结论。
2. 消除重复信息。
3. 如果子答案指出信息不足，最终答案也要说明。
4. 直接输出最终答案。
"""
    prompt += """

补充要求：
1. 只输出最终结论，不要输出“具体分析如下”“子问题”“信息补充说明”等过程性内容。
2. 不要添加来源章节，来源会由系统统一补充。
3. 直接回答原始问题。"""
    return await llm.generate(prompt=prompt, max_tokens=1200)


@router.post("/query/enhanced", response_model=EnhancedQueryResponse)
async def enhanced_query(request: EnhancedQueryRequest) -> EnhancedQueryResponse | StreamingResponse:
    try:
        understanding = await QueryUnderstanding().understand_query(
            query=request.query,
            chat_history=request.chat_history,
            enable_coreference=request.enable_coreference,
            enable_decomposition=request.enable_decomposition,
            enable_rewrite=request.enable_rewrite,
        )

        engine = HybridRetrievalEngine()
        generator = Generator()
        subqueries = understanding["subqueries"]

        # 检索用"原查询 + 改写/拆分查询"全集，提升召回；并行检索后合并去重。
        retrieval_queries = understanding["retrieval_queries"]
        retrieval_tasks = [
            engine.retrieve(query=rq, top_k=request.top_k, partition=request.partition)
            for rq in retrieval_queries
        ]
        result_groups = await asyncio.gather(*retrieval_tasks)
        merged_results = _merge_results(result_groups, request.top_k, retrieval_queries)

        # query -> docs 映射，便于子答案按子问题取回各自结果
        # （result_groups 对应 retrieval_queries，与 subqueries 数量/顺序不一致）。
        query_to_docs = dict(zip(retrieval_queries, result_groups, strict=False))

        if request.stream:
            async def stream_generator():
                yield f"data: {json.dumps({'type': 'understanding', 'data': understanding}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'data': merged_results}, ensure_ascii=False)}\n\n"

                if understanding["is_decomposed"]:
                    sub_docs = [query_to_docs.get(sq, merged_results) for sq in subqueries]
                    answer_tasks = [
                        generator.generate(query=subquery, context_docs=docs, use_cache=False)
                        for subquery, docs in zip(subqueries, sub_docs, strict=False)
                    ]
                    answers = await asyncio.gather(*answer_tasks)
                    stream_sub_answers = [
                        {"query": subquery, "answer": answer, "result_count": len(docs)}
                        for subquery, answer, docs in zip(subqueries, answers, sub_docs, strict=False)
                    ]
                    answer = await _aggregate_sub_answers(understanding["resolved_query"], stream_sub_answers)
                    answer = generator._format_answer(answer, merged_results)
                    yield f"data: {json.dumps({'type': 'chunk', 'data': answer}, ensure_ascii=False)}\n\n"
                else:
                    answer = await generator.generate(
                        query=understanding["resolved_query"],
                        context_docs=merged_results,
                    )
                    for chunk in _stream_text_chunks(answer):
                        yield f"data: {json.dumps({'type': 'chunk', 'data': chunk}, ensure_ascii=False)}\n\n"

                if request.enable_evaluation:
                    evaluation = await LLMJudge().evaluate_rag_pipeline(
                        query=understanding["resolved_query"],
                        contexts=merged_results,
                        answer=answer,
                    )
                    yield f"data: {json.dumps({'type': 'evaluation', 'data': evaluation}, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'total_time': time.time()}, ensure_ascii=False)}\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        sub_answers: list[dict[str, Any]] = []
        if understanding["is_decomposed"]:
            sub_docs = [query_to_docs.get(sq, merged_results) for sq in subqueries]
            answer_tasks = [
                generator.generate(query=subquery, context_docs=docs, use_cache=False)
                for subquery, docs in zip(subqueries, sub_docs, strict=False)
            ]
            answers = await asyncio.gather(*answer_tasks)
            sub_answers = [
                {"query": subquery, "answer": answer, "result_count": len(docs)}
                for subquery, answer, docs in zip(subqueries, answers, sub_docs, strict=False)
            ]
            answer = await _aggregate_sub_answers(understanding["resolved_query"], sub_answers)
            answer = generator._format_answer(answer, merged_results)
        else:
            answer = await generator.generate(
                query=understanding["resolved_query"],
                context_docs=merged_results,
            )

        evaluation = None
        if request.enable_evaluation:
            evaluation = await LLMJudge().evaluate_rag_pipeline(
                query=understanding["resolved_query"],
                contexts=merged_results,
                answer=answer,
            )

        return EnhancedQueryResponse(
            query=request.query,
            understanding=understanding,
            results=merged_results,
            answer=answer,
            sub_answers=sub_answers,
            evaluation=evaluation,
        )
    except Exception as exc:
        logger.error("Enhanced query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class EvaluationRequest(BaseModel):
    query: str
    contexts: list[dict[str, Any]]
    answer: str
    ground_truth: str | None = None


class EvaluationResponse(BaseModel):
    retrieval: dict[str, Any]
    generation: dict[str, Any]
    ragas_metrics: dict[str, float] | None = None


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest) -> EvaluationResponse:
    try:
        pipeline_eval = await LLMJudge().evaluate_rag_pipeline(
            query=request.query,
            contexts=request.contexts,
            answer=request.answer,
        )
        ragas_metrics = None
        if request.ground_truth:
            ragas_metrics = await RAGASEvaluator().compute_ragas_metrics(
                query=request.query,
                contexts=request.contexts,
                answer=request.answer,
                ground_truth=request.ground_truth,
            )
        return EvaluationResponse(
            retrieval=pipeline_eval["retrieval"],
            generation=pipeline_eval["generation"],
            ragas_metrics=ragas_metrics,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
