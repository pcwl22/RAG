"""Service orchestration tests."""
import asyncio

import pytest

import app.service.enhanced_query_service as enhanced_service
import app.service.ingest_service as ingest_service
from app.service.chat_service import Generator, _doc_debug_summary


def test_unstructured_aggregate_prompt_does_not_require_json_contract():
    prompt = enhanced_service.build_aggregate_prompt(
        "原始问题",
        [{"query": "子问题", "answer": "子答案"}],
        context="可核验上下文",
        structured=False,
    )

    assert "直接输出最终答案正文" in prompt
    assert '"conclusion"' not in prompt


def test_document_debug_summary_does_not_log_legal_text_or_source_names():
    summary = _doc_debug_summary(
        [
            {
                "id": "private-chunk",
                "score": 0.9,
                "content": "受保护的法律正文不应进入日志",
                "metadata": {"filename": "private-case.docx", "chunk_index": 2},
            }
        ]
    )

    serialized = str(summary)
    assert "受保护的法律正文" not in serialized
    assert "private-case.docx" not in serialized
    assert "preview" not in summary[0]
    assert summary[0]["content_length"] > 0


def test_generator_formats_answer_and_streams_basis():
    async def run():
        class FakeLLM:
            async def generate_stream(self, **kwargs):
                yield "stream chunk"

        generator = Generator.__new__(Generator)
        generator.config = {
            "context_template": "Doc {index}\nSource: {source}\nScore: {score:.2f}\n{content}\n",
            "system_prompt": "system",
            "temperature": 0.1,
            "max_tokens": 64,
        }
        generator.llm = FakeLLM()

        docs = [
            {
                "id": "chunk1",
                "score": 0.91,
                "content": "matched source text",
                "metadata": {"filename": "demo.txt", "chunk_index": 0, "page": 1},
            }
        ]

        formatted = generator.format_answer("fake answer body", docs)
        stream_text = "".join([chunk async for chunk in generator.generate_stream("question", docs)])

        assert "【回答】" in formatted
        assert "结论" in formatted
        assert "依据" in formatted
        assert "demo.txt" in formatted
        assert "stream chunk" in stream_text
        assert "依据" in stream_text

    asyncio.run(run())


def test_generator_omits_irrelevant_basis_when_answer_declines():
    async def run():
        class FakeLLM:
            async def generate_stream(self, **kwargs):
                yield "无法基于当前知识库回答。检索到的上下文均未涉及该问题，需要补充资料。"

        generator = Generator.__new__(Generator)
        generator.config = {
            "context_template": "Doc {index}\nSource: {source}\nScore: {score:.2f}\n{content}\n",
            "system_prompt": "system",
            "temperature": 0.1,
            "max_tokens": 64,
        }
        generator.llm = FakeLLM()

        docs = [
            {
                "id": "irrelevant",
                "score": 0.91,
                "content": "第二条 中华人民共和国刑法的任务，是用刑罚同一切犯罪行为作斗争。",
                "metadata": {"filename": "中华人民共和国刑法_20201226.docx", "chunk_index": 0},
            }
        ]
        decline = "无法基于当前知识库回答。检索到的上下文均未涉及该问题，需要补充资料。"

        formatted = generator.format_answer(decline, docs)
        stream_text = "".join([chunk async for chunk in generator.generate_stream("question", docs)])

        assert "无直接支持结论的命中文档" in formatted
        assert "无直接支持结论的命中文档" in stream_text
        assert "第二条 中华人民共和国刑法" not in formatted
        assert "第二条 中华人民共和国刑法" not in stream_text

    asyncio.run(run())


def test_generator_filters_basis_to_answer_evidence():
    generator = Generator.__new__(Generator)
    docs = [
        {
            "id": "drug-definition",
            "score": 0.9,
            "content": "第三百五十七条 本法所称的毒品，是指国家规定管制的其他能够使人形成瘾癖的麻醉药品和精神药品。",
            "metadata": {"filename": "中华人民共和国刑法_20201226.docx"},
        },
        {
            "id": "fine",
            "score": 0.8,
            "content": "第五十二条 判处罚金，应当根据犯罪情节决定罚金数额。",
            "metadata": {"filename": "中华人民共和国刑法_20201226.docx"},
        },
    ]

    answer = "多次贩卖含依托咪酯的上头电子烟，符合条件的，按贩卖毒品罪处理。"
    formatted = generator.format_answer(answer, docs, "多次贩卖含依托咪酯的上头电子烟，如何定罪？")

    assert "第三百五十七条" in formatted
    assert "第五十二条" not in formatted


def test_generator_basis_keeps_priority_mapped_articles():
    generator = Generator.__new__(Generator)
    docs = [
        {
            "id": "劳动合同法_总则_2条",
            "score": 0.9,
            "mapped_article_priority": 3,
            "content": "第二条 用人单位与劳动者建立劳动关系，适用本法。",
            "metadata": {"filename": "中华人民共和国劳动合同法_20121228.docx"},
        },
        {
            "id": "劳动合同法_劳动合同的订立_7条",
            "score": 0.8,
            "mapped_article_priority": 2,
            "content": "第七条 用人单位自用工之日起即与劳动者建立劳动关系。",
            "metadata": {"filename": "中华人民共和国劳动合同法_20121228.docx"},
        },
        {
            "id": "劳动合同法_劳动合同的订立_10条",
            "score": 0.7,
            "content": "第十条 建立劳动关系，应当订立书面劳动合同。",
            "metadata": {"filename": "中华人民共和国劳动合同法_20121228.docx"},
        },
    ]

    answer = "劳动者应首先向建立劳动关系的用人单位主张权利。"
    formatted = generator.format_answer(answer, docs, "关联企业混同用工，劳动者该向谁主张权利？")

    assert "第二条" in formatted
    assert "第七条" in formatted
    assert formatted.index("第二条") < formatted.index("第七条")


def test_enhanced_query_refuses_empty_retrieval_without_calling_llm():
    async def run():
        understanding = {
            "original_query": "知识库外问题",
            "resolved_query": "知识库外问题",
            "rewritten_query": "知识库外问题",
            "subqueries": ["知识库外问题"],
            "is_decomposed": False,
            "retrieval_queries": ["知识库外问题"],
            "concept_article_mappings": [],
        }

        class FakeUnderstanding:
            async def understand_query(self, **kwargs):
                return dict(understanding)

        class EmptyRetrieval:
            async def retrieve(self, *args, **kwargs):
                return []

        class FailGenerator:
            async def generate(self, *args, **kwargs):
                raise AssertionError("LLM generation must not run without evidence")

            async def generate_stream(self, *args, **kwargs):
                raise AssertionError("LLM streaming must not run without evidence")
                yield  # pragma: no cover

        service = enhanced_service.EnhancedQueryService(
            query_understanding=FakeUnderstanding(),
            retrieval_engine=EmptyRetrieval(),
            generator=FailGenerator(),
        )
        options = enhanced_service.EnhancedQueryOptions(query="知识库外问题")

        result = await service.run(options)
        events = [event async for event in service.stream_events(options)]

        assert result.results == []
        assert "无法基于当前知识库可靠回答" in result.answer
        assert any(
            event["type"] == "chunk" and "无法基于当前知识库可靠回答" in event["data"]
            for event in events
        )

    asyncio.run(run())


def test_enhanced_retrieval_rejects_known_out_of_scope_before_rewrite_results():
    async def run():
        understanding = {
            "resolved_query": "发明专利优先审查条件具体如何规定？",
            "retrieval_queries": [
                "发明专利优先审查条件具体如何规定？",
                "专利审查程序",
            ],
        }

        class FailRetrieval:
            async def retrieve(self, *args, **kwargs):
                raise AssertionError("out-of-scope enhanced queries must not retrieve")

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=FailRetrieval(), generator=object()
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert results == []
        assert query_to_docs == {query: [] for query in understanding["retrieval_queries"]}
        assert understanding["out_of_scope"] is True

    asyncio.run(run())


def test_generator_keeps_basis_for_partial_supported_conclusion():
    generator = Generator.__new__(Generator)
    docs = [
        {
            "id": "劳动合同法_劳动合同的订立_7条",
            "score": 0.8,
            "mapped_article_priority": 1,
            "content": "第七条 用人单位自用工之日起即与劳动者建立劳动关系。",
            "metadata": {"filename": "中华人民共和国劳动合同法_20121228.docx"},
        }
    ]
    answer = (
        "劳动者应首先向与其存在实际用工关系的用人单位主张权利。"
        "当前上下文未涉及关联企业连带责任规则，需要补充专门资料。"
    )

    formatted = generator.format_answer(answer, docs, "关联企业混同用工，劳动者该向谁主张权利？")

    assert "无直接支持结论的命中文档" not in formatted
    assert "第七条" in formatted


def test_generator_context_prefers_child_content_and_caps_each_doc():
    generator = Generator.__new__(Generator)
    generator.config = {
        "context_template": "Doc {index}\nSource: {source}\nScore: {score:.2f}\n{content}\n",
        "max_context_per_doc": 80,
    }

    context = generator._build_context(
        [
            {
                "id": "criminal-law-child",
                "score": 0.9,
                "rrf_score": 9.0,
                "content": "parent content " * 200,
                "child_content": "第二百七十一条 利用职务上的便利，将本单位财物非法占为己有。",
                "metadata": {"filename": "criminal-law.docx", "rerank_prob": 0.77},
            },
            {
                "id": "long-child",
                "score": 0.8,
                "child_content": "第二百六十四条 " + ("盗窃公私财物 " * 50),
                "metadata": {"filename": "criminal-law.docx"},
            },
        ]
    )

    assert "第二百七十一条" in context
    assert "parent content" not in context
    assert "Score: 0.77" in context
    assert "..." in context


def test_generator_refuses_when_requested_law_is_missing():
    async def run():
        generator = Generator.__new__(Generator)
        generator.config = {}

        class FakeLLM:
            async def generate(self, **kwargs):
                raise AssertionError("LLM should not be called for missing requested law")

        generator.llm = FakeLLM()
        generator.cache = None

        answer = await generator.generate(
            "（公司法） 公司为股东提供担保是否需要股东会决议？",
            [
                {
                    "id": "civil-code",
                    "score": 0.2,
                    "content": "unrelated civil code content",
                    "metadata": {"filename": "中华人民共和国民法典_20200528.docx"},
                }
            ],
        )

        assert "当前知识库未检索到《公司法》相关文档" in answer
        assert "无匹配的请求法律文档" in answer

    asyncio.run(run())


def test_enhanced_stream_uses_aggregate_answer_for_basis_filter(monkeypatch):
    async def run():
        understanding = {
            "original_query": "compound",
            "resolved_query": "compound resolved",
            "rewritten_query": "compound rewritten",
            "retrieval_signals": {},
            "subqueries": ["sub one", "sub two"],
            "retrieval_queries": ["compound resolved", "sub one", "sub two"],
            "is_decomposed": True,
        }

        class FakeUnderstanding:
            async def understand_query(self, **kwargs):
                return understanding

        class FakeEngine:
            async def retrieve(self, query, **kwargs):
                return [
                    {
                        "id": query,
                        "score": 0.9,
                        "content": f"source for {query}",
                        "metadata": {"filename": f"{query}.txt"},
                    }
                ]

        class FakeGenerator:
            def __init__(self):
                self.basis_answer = None
                self.basis_query = None

            async def generate(self, query, context_docs, use_cache=True):
                return f"answer for {query}"

            def format_basis_section(self, context_docs, answer=None, query=None):
                self.basis_answer = answer
                self.basis_query = query
                return "\n\n依据：ok"

        aggregate_context = None

        async def fake_aggregate_stream(query, sub_answers, context=""):
            nonlocal aggregate_context
            aggregate_context = context
            yield "最终"
            yield "答案"

        fake_generator = FakeGenerator()
        monkeypatch.setattr(
            enhanced_service,
            "aggregate_sub_answers_stream",
            fake_aggregate_stream,
        )
        service = enhanced_service.EnhancedQueryService(
            query_understanding=FakeUnderstanding(),
            retrieval_engine=FakeEngine(),
            generator=fake_generator,
        )

        events = [
            event
            async for event in service.stream_events(
                enhanced_service.EnhancedQueryOptions(query="compound")
            )
        ]

        assert fake_generator.basis_answer == "最终答案"
        assert fake_generator.basis_query == "compound resolved"
        assert "source for compound resolved" in aggregate_context
        assert events[-1] == {"type": "chunk", "data": "\n\n依据：ok"}

    asyncio.run(run())


def test_enhanced_retrieval_expands_and_prioritizes_mapped_articles():
    async def run():
        understanding = {
            "resolved_query": "关联企业混同用工，劳动者该向谁主张权利？",
            "retrieval_queries": ["plain query", "mapped query"],
            "concept_article_mappings": [
                {
                    "articles": [
                        {
                            "semantic_chunk_id": "劳动合同法_劳动合同的订立_7条",
                        }
                    ]
                }
            ],
        }

        class FakeEngine:
            def __init__(self):
                self.top_k_calls = []
                self.exact_id_calls = []

            async def retrieve(self, query, top_k, **kwargs):
                self.top_k_calls.append(top_k)
                return [
                    {
                        "id": f"{query}-noise",
                        "score": 0.99,
                        "content": "weakly related high score",
                        "metadata": {},
                    }
                ]

            async def retrieve_by_ids(self, ids, partition=None):
                self.exact_id_calls.append((ids, partition))
                return [
                    {
                        "id": "劳动合同法_劳动合同的订立_7条",
                        "score": 0.2,
                        "content": "第七条 用人单位自用工之日起即与劳动者建立劳动关系。",
                        "metadata": {"semantic_chunk_id": "劳动合同法_劳动合同的订立_7条"},
                    },
                ]

        fake_engine = FakeEngine()
        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(),
            retrieval_engine=fake_engine,
            generator=object(),
        )

        results, _ = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(
                query=understanding["resolved_query"],
                top_k=1,
            ),
        )

        assert fake_engine.top_k_calls == [10, 10]
        assert fake_engine.exact_id_calls == [(["劳动合同法_劳动合同的订立_7条"], None)]
        assert [doc["id"] for doc in results] == ["劳动合同法_劳动合同的订立_7条"]

    asyncio.run(run())


def test_coerced_accomplice_keeps_common_crime_context():
    docs = [
        {
            "id": "刑法_总则_犯罪_共同犯罪_28条",
            "content": "第二十八条 对于被胁迫参加犯罪的，应当减轻处罚或者免除处罚。",
            "metadata": {"article_number": "第二十八条"},
        }
    ]

    assert enhanced_service.filter_untriggered_context_docs(
        "受胁迫参加犯罪的胁从犯依法应如何处罚？", docs
    ) == docs


def test_explicit_citation_bypasses_fact_trigger_filter():
    docs = [
        {
            "id": "刑法_总则_犯罪_共同犯罪_29条",
            "content": "第二十九条 教唆他人犯罪的，应当按照共同犯罪处理。",
            "explicit_citation_exact_match": True,
            "metadata": {"article_number": "第二十九条"},
        }
    ]
    assert enhanced_service.filter_untriggered_context_docs(
        "请说明《中华人民共和国刑法》第二十九条规定的主要内容。", docs
    ) == docs


def test_mapped_article_exact_match_bypasses_fact_trigger_filter():
    docs = [
        {
            "id": "刑法_总则_犯罪_共同犯罪_28条",
            "content": "第二十八条 对于被胁迫参加犯罪的，应当减轻处罚或者免除处罚。",
            "mapped_article_exact_match": True,
            "metadata": {"article_number": "第二十八条"},
        }
    ]

    assert enhanced_service.filter_untriggered_context_docs("一般合同纠纷如何处理？", docs) == docs


def test_decomposed_subanswers_receive_controlled_priority_context():
    async def run():
        calls = []

        class FakeGenerator:
            async def generate(self, query, context_docs, use_cache=False):
                calls.append((query, context_docs))
                return "sub-answer"

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=object(), generator=FakeGenerator()
        )
        priority = {
            "id": "mapped",
            "content": "controlled article",
            "mapped_article_priority": 1,
            "metadata": {"semantic_chunk_id": "mapped-id"},
        }
        noise = {"id": "noise", "content": "subquery-specific context", "metadata": {}}

        answers = await service.generate_sub_answers(
            ["subquery"],
            {"subquery": [noise]},
            [priority],
            priority_docs=[priority],
        )

        assert answers[0]["result_count"] == 2
        assert [doc["id"] for doc in calls[0][1]] == ["mapped", "noise"]

    asyncio.run(run())


def test_enhanced_retrieval_preserves_explicit_citation_exact_match():
    async def run():
        understanding = {
            "resolved_query": "对比《中华人民共和国刑法》第一条与第二条",
            "retrieval_queries": ["rewritten query"],
            "concept_article_mappings": [],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {"id": "noise", "content": "noise", "score": 0.99, "metadata": {}},
                    {
                        "id": "exact",
                        "content": "第一条",
                        "score": 1.0,
                        "metadata": {"semantic_chunk_id": "exact"},
                        "explicit_citation_exact_match": True,
                    },
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=FakeEngine(), generator=object()
        )
        results, _ = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"], top_k=2),
        )
        assert [doc["id"] for doc in results] == ["exact"]

    asyncio.run(run())


def test_dynamic_context_selection_drops_flat_score_noise_but_keeps_mapped_articles():
    docs = [
        {
            "id": "noise-1",
            "score": 0.997,
            "content": "第十一条 劳动报酬约定不明确",
            "metadata": {},
        },
        {
            "id": "劳动合同法_法律责任_82条",
            "score": 0.995,
            "content": "第八十二条 未订立书面劳动合同的二倍工资责任",
            "metadata": {"semantic_chunk_id": "劳动合同法_法律责任_82条"},
        },
        {
            "id": "noise-2",
            "score": 0.992,
            "content": "第三十六条 协商一致解除",
            "metadata": {},
        },
    ]

    selected = enhanced_service.select_dynamic_context_docs(
        docs,
        top_k=3,
        priority_ids=["劳动合同法_法律责任_82条"],
    )

    assert [doc["id"] for doc in selected] == ["劳动合同法_法律责任_82条"]
    assert selected[0]["context_selection"] == "mapped_article"


def test_dynamic_context_selection_uses_score_elbow_without_mapping():
    docs = [
        {"id": "a", "score": 0.94},
        {"id": "b", "score": 0.90},
        {"id": "c", "score": 0.70},
        {"id": "d", "score": 0.69},
    ]

    selected = enhanced_service.select_dynamic_context_docs(docs, top_k=4)

    assert [doc["id"] for doc in selected] == ["a", "b"]


def test_dynamic_context_selection_preserves_strong_multi_query_evidence():
    docs = [
        {
            "id": "specific-match",
            "score": 0.42,
            "source_relevance_score": 0.91,
        },
        {"id": "broad-match", "score": 0.80, "source_relevance_score": 0.80},
    ]

    selected = enhanced_service.select_dynamic_context_docs(docs, top_k=1)

    assert [doc["id"] for doc in selected] == ["specific-match"]
    assert selected[0]["context_selection_score"] == pytest.approx(0.812)


def test_dynamic_context_selection_uses_common_query_to_correct_saturated_rewrite():
    docs = [
        {
            "id": "unrelated-rewrite-match",
            "score": 0.62,
            "source_relevance_score": 0.998,
        },
        {
            "id": "supported-original-match",
            "score": 0.935,
            "source_relevance_score": 0.992,
        },
    ]

    selected = enhanced_service.select_dynamic_context_docs(docs, top_k=1)

    assert [doc["id"] for doc in selected] == ["supported-original-match"]


def test_dynamic_context_selection_preserves_each_queries_top_candidate():
    docs = [
        {
            "id": "rewrite-favorite",
            "score": 0.99,
            "source_relevance_score": 0.99,
            "matched_queries": ["original", "rewrite"],
            "query_retrieval_ranks": {"original": 8, "rewrite": 1},
        },
        {
            "id": "original-favorite",
            "score": 0.93,
            "source_relevance_score": 0.35,
            "matched_queries": ["original"],
            "query_retrieval_ranks": {"original": 1},
        },
    ]

    selected = enhanced_service.select_dynamic_context_docs(
        docs,
        top_k=2,
        coverage_queries=["original", "rewrite"],
    )

    assert [doc["id"] for doc in selected] == [
        "original-favorite",
        "rewrite-favorite",
    ]
    assert all(doc["context_selection"] == "query_coverage" for doc in selected)


def test_context_coverage_queries_excludes_internal_signal_queries():
    understanding = {
        "resolved_query": "original",
        "subqueries": ["rewrite one", "rewrite two"],
        "is_decomposed": True,
        "retrieval_queries": [
            "original",
            "rewrite one",
            "rewrite two",
            "synthetic signal query",
            "controlled article query",
        ],
    }

    assert enhanced_service.context_coverage_queries(understanding) == [
        "original",
        "rewrite one",
        "rewrite two",
    ]

    understanding["is_decomposed"] = False
    assert enhanced_service.context_coverage_queries(understanding) == ["original"]


def test_dynamic_context_selection_retains_one_multi_query_consensus_candidate():
    drifted_original = {
        "id": "drifted-original-top",
        "score": 0.99,
        "source_relevance_score": 0.56,
        "matched_queries": ["original"],
        "query_retrieval_ranks": {"original": 1},
    }
    consensus_target = {
        "id": "consensus-target",
        "score": 0.99,
        "source_relevance_score": 0.72,
        "matched_queries": ["original", "rewrite"],
        "query_retrieval_ranks": {"original": 6, "rewrite": 5},
    }
    rewrite_neighbors = [
        {
            "id": f"rewrite-neighbor-{index}",
            "score": score,
            "source_relevance_score": score,
            "matched_queries": ["rewrite"],
            "query_retrieval_ranks": {"rewrite": index},
        }
        for index, score in enumerate((0.95, 0.94, 0.93, 0.92), 1)
    ]

    selected = enhanced_service.select_dynamic_context_docs(
        [drifted_original, consensus_target, *rewrite_neighbors],
        top_k=5,
        coverage_queries=["original"],
    )

    assert selected[0]["id"] == "drifted-original-top"
    assert selected[1]["id"] == "consensus-target"
    assert selected[1]["context_selection"] == "multi_query_consensus"


def test_enhanced_retrieval_requests_wide_rerank_candidate_budget(monkeypatch):
    async def run():
        calls = []

        class FakeEngine:
            async def retrieve(self, **kwargs):
                calls.append(kwargs)
                return []

        monkeypatch.setattr(
            enhanced_service,
            "get_settings",
            lambda: {"rag": {"retrieval": {"merge_candidate_multiplier": 8}}},
        )
        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=FakeEngine(), generator=object()
        )
        understanding = {
            "resolved_query": "劳动合同解除条件",
            "retrieval_queries": ["劳动合同解除条件", "劳动合同解除"],
            "concept_article_mappings": [],
        }

        await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(
                query=understanding["resolved_query"], top_k=5, enable_rerank=True
            ),
        )

        assert len(calls) == 2
        assert {call["rerank_top_k"] for call in calls} == {40}

    asyncio.run(run())


def test_enhanced_retrieval_keeps_all_explicit_citations_above_top_k():
    async def run():
        understanding = {
            "resolved_query": "比较民法典第一条与第二条",
            "retrieval_queries": ["比较民法典第一条与第二条"],
            "concept_article_mappings": [],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": f"exact-{article}",
                        "content": article,
                        "score": 1.0,
                        "explicit_citation_exact_match": True,
                        "metadata": {
                            "semantic_chunk_id": f"exact-{article}",
                            "law_name": "中华人民共和国民法典",
                            "article_number": article,
                        },
                    }
                    for article in ("第一条", "第二条")
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=FakeEngine(), generator=object()
        )
        results, _ = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(
                query=understanding["resolved_query"], top_k=1, enable_rerank=False
            ),
        )

        assert [doc["id"] for doc in results] == ["exact-第一条", "exact-第二条"]

    asyncio.run(run())


def test_mapped_article_coverage_reports_missing_corpus_targets():
    coverage = enhanced_service.mapped_article_coverage(
        ["劳动合同法_法律责任_82条", "劳动合同法实施条例_劳动合同的订立_6条"],
        [
            {
                "id": "劳动合同法_法律责任_82条",
                "metadata": {"semantic_chunk_id": "劳动合同法_法律责任_82条"},
            }
        ],
    )

    assert coverage["coverage"] == 0.5
    assert coverage["missing_ids"] == ["劳动合同法实施条例_劳动合同的订立_6条"]


def test_enhanced_retrieval_filters_untriggered_common_crime_context():
    async def run():
        understanding = {
            "resolved_query": "假借废品回收便利长期收赃销赃，应如何惩处？",
            "retrieval_queries": ["收赃销赃"],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": "刑法_312条",
                        "score": 0.9,
                        "content": "第三百一十二条 明知是犯罪所得而予以收购、代为销售。",
                        "metadata": {},
                    },
                    {
                        "id": "刑法_310条",
                        "score": 0.8,
                        "content": "第三百一十条 犯前款罪，事前通谋的，以共同犯罪论处。",
                        "metadata": {},
                    },
                    {
                        "id": "刑法_191条",
                        "score": 0.7,
                        "content": "第一百九十一条 为掩饰、隐瞒毒品犯罪等所得来源和性质。",
                        "metadata": {},
                    },
                    {
                        "id": "刑法_313条",
                        "score": 0.6,
                        "content": "第三百一十三条 对人民法院的判决、裁定有能力执行而拒不执行。",
                        "metadata": {},
                    },
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(),
            retrieval_engine=FakeEngine(),
            generator=object(),
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert [doc["id"] for doc in results] == ["刑法_312条"]
        assert [doc["id"] for doc in query_to_docs["收赃销赃"]] == ["刑法_312条"]

    asyncio.run(run())


def test_enhanced_retrieval_filters_employee_remedy_context_for_employer_dismissal():
    async def run():
        understanding = {
            "resolved_query": "公司让员工签订空白劳动合同后又以此辞退，是否合法？",
            "retrieval_queries": ["空白劳动合同 辞退 合法性"],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": "劳动合同法_38条",
                        "score": 0.9,
                        "content": "第三十八条 用人单位未依法为劳动者缴纳社会保险费的，劳动者可以解除劳动合同。",
                        "metadata": {},
                    },
                    {
                        "id": "劳动合同法_46条",
                        "score": 0.8,
                        "content": "第四十六条 劳动者依照本法第三十八条规定解除劳动合同的，用人单位应当支付经济补偿。",
                        "metadata": {},
                    },
                    {
                        "id": "劳动合同法_39条",
                        "score": 0.7,
                        "content": "第三十九条 劳动者有下列情形之一的，用人单位可以解除劳动合同。",
                        "metadata": {},
                    },
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(),
            retrieval_engine=FakeEngine(),
            generator=object(),
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert [doc["id"] for doc in results] == ["劳动合同法_39条"]
        assert [doc["id"] for doc in query_to_docs["空白劳动合同 辞退 合法性"]] == [
            "劳动合同法_39条"
        ]

    asyncio.run(run())


def test_enhanced_retrieval_keeps_employee_remedy_context_when_query_triggers_it():
    async def run():
        understanding = {
            "resolved_query": "劳动者承诺自愿放弃社保后，还能以未缴社保为由要求经济补偿吗？",
            "retrieval_queries": ["未缴社保 经济补偿"],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": "劳动合同法_38条",
                        "score": 0.9,
                        "content": "第三十八条 用人单位未依法为劳动者缴纳社会保险费的，劳动者可以解除劳动合同。",
                        "metadata": {},
                    },
                    {
                        "id": "劳动合同法_46条",
                        "score": 0.8,
                        "content": "第四十六条 劳动者依照本法第三十八条规定解除劳动合同的，用人单位应当支付经济补偿。",
                        "metadata": {},
                    },
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(),
            retrieval_engine=FakeEngine(),
            generator=object(),
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert [doc["id"] for doc in results] == ["劳动合同法_38条", "劳动合同法_46条"]
        assert [doc["id"] for doc in query_to_docs["未缴社保 经济补偿"]] == [
            "劳动合同法_38条",
            "劳动合同法_46条",
        ]

    asyncio.run(run())


def test_enhanced_retrieval_keeps_common_crime_context_when_query_triggers_it():
    async def run():
        understanding = {
            "resolved_query": "收赃人与盗窃人事前通谋，应如何处理？",
            "retrieval_queries": ["事前通谋 共同犯罪"],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": "刑法_310条",
                        "score": 0.8,
                        "content": "第三百一十条 犯前款罪，事前通谋的，以共同犯罪论处。",
                        "metadata": {},
                    }
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(),
            retrieval_engine=FakeEngine(),
            generator=object(),
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert [doc["id"] for doc in results] == ["刑法_310条"]
        assert [doc["id"] for doc in query_to_docs["事前通谋 共同犯罪"]] == ["刑法_310条"]

    asyncio.run(run())


def test_enhanced_retrieval_keeps_common_crime_context_for_incitement():
    async def run():
        understanding = {
            "resolved_query": "甲教唆十五岁的乙盗窃，但乙没有实施。对甲应如何处理？",
            "retrieval_queries": ["教唆未成年人犯罪 被教唆人未实施犯罪"],
        }

        class FakeEngine:
            async def retrieve(self, **kwargs):
                return [
                    {
                        "id": "刑法_29条",
                        "score": 0.9,
                        "content": "第二十九条 教唆他人犯罪的，应当按照他在共同犯罪中所起的作用处罚。",
                        "metadata": {},
                    }
                ]

        service = enhanced_service.EnhancedQueryService(
            query_understanding=object(), retrieval_engine=FakeEngine(), generator=object()
        )

        results, query_to_docs = await service.retrieve(
            understanding,
            enhanced_service.EnhancedQueryOptions(query=understanding["resolved_query"]),
        )

        assert [doc["id"] for doc in results] == ["刑法_29条"]
        assert [
            doc["id"]
            for doc in query_to_docs["教唆未成年人犯罪 被教唆人未实施犯罪"]
        ] == ["刑法_29条"]

    asyncio.run(run())


def test_process_document_structures_legal_articles(monkeypatch):
    async def run():
        sample_text = """
中华人民共和国测试法

第一编 总则
第一章 基本规定
第一节 一般规定
第一条 为了测试结构化入库，制定本法。
第二条 测试行为造成结果的，应当依法处理。
""".strip()

        async def fake_replace_document(
            source_key, filename, ids, embeddings, documents, metadatas, partition="general"
        ):
            fake_replace_document.calls = {
                "source_key": source_key,
                "filename": filename,
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
                "partition": partition,
            }
            return len(ids)

        monkeypatch.setattr(ingest_service, "parse_document", lambda file_path: sample_text)
        monkeypatch.setattr(
            ingest_service,
            "encode_texts",
            lambda texts, batch_size=32: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(ingest_service, "_storage_backend", lambda: fake_replace_document)

        result = await ingest_service.process_document("unused.docx", "中华人民共和国测试法_20240101.docx")

        metadatas = fake_replace_document.calls["metadatas"]
        assert result["status"] == "completed"
        assert result["total_chunks"] == 2
        assert fake_replace_document.calls["documents"][0].startswith("第一条")
        assert metadatas[0]["document_type"] == "legal_article"
        assert metadatas[0]["law_name"] == "中华人民共和国测试法"
        assert metadatas[0]["law_chapter"] == "第一章 基本规定"
        assert metadatas[0]["article_number"] == "第一条"
        assert metadatas[0]["article_text"] == fake_replace_document.calls["documents"][0]
        assert fake_replace_document.calls["ids"][0] != metadatas[0]["semantic_chunk_id"]
        assert metadatas[0]["chunk_id"] == fake_replace_document.calls["ids"][0]
        assert metadatas[0]["source_key"] == fake_replace_document.calls["source_key"]

    asyncio.run(run())


def test_process_document_orchestrates_parser_embedding_and_storage(monkeypatch):
    async def run():
        sample_text = ("Article 1. Module smoke test content.\n\n" * 20).strip()

        async def fake_replace_document(
            source_key, filename, ids, embeddings, documents, metadatas, partition="general"
        ):
            fake_replace_document.calls = {
                "source_key": source_key,
                "filename": filename,
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
                "partition": partition,
            }
            return len(ids)

        monkeypatch.setattr(ingest_service, "parse_document", lambda file_path: sample_text)
        monkeypatch.setattr(
            ingest_service,
            "chunk_text_parent_child",
            lambda text: {
                "child_chunks": [
                    {
                        "child_id": "child-1",
                        "child_content": "child content one " * 80,
                        "parent_id": "parent-1",
                        "parent_content": "parent content one",
                        "child_index": 0,
                        "total_children": 2,
                    },
                    {
                        "child_id": "child-2",
                        "child_content": "child content two " * 80,
                        "parent_id": "parent-1",
                        "parent_content": "parent content one",
                        "child_index": 1,
                        "total_children": 2,
                    },
                ]
            },
        )
        monkeypatch.setattr(
            ingest_service,
            "encode_texts",
            lambda texts, batch_size=32: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(ingest_service, "_storage_backend", lambda: fake_replace_document)

        result = await ingest_service.process_document("unused.txt", "demo.txt")

        assert result["status"] == "completed"
        assert result["total_chunks"] == 2
        assert len(fake_replace_document.calls["ids"]) == 2
        assert fake_replace_document.calls["metadatas"][0]["filename"] == "demo.txt"
        assert fake_replace_document.calls["filename"] == "demo.txt"

    asyncio.run(run())


def test_process_document_does_not_trust_parser_owned_upload_metadata(monkeypatch):
    async def run():
        sample_text = ("Plain document content for metadata isolation tests.\n\n" * 20).strip()

        async def fake_replace_document(
            source_key, filename, ids, embeddings, documents, metadatas, partition="general"
        ):
            fake_replace_document.calls = metadatas
            return len(ids)

        monkeypatch.setattr(ingest_service, "parse_document", lambda file_path: sample_text)
        monkeypatch.setattr(
            ingest_service,
            "encode_texts",
            lambda texts, batch_size=32: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(ingest_service, "_storage_backend", lambda: fake_replace_document)

        result = await ingest_service.process_document(
            "unused.txt",
            "actual.txt",
            metadata={
                "source_id": "stable-source",
                "authority": "official",
                "tenant_id": "00000000-0000-0000-0000-000000000099",
                "document_id": "forged-document",
                "upload_task_id": "forged-task",
                "upload_object_key": "forged-object",
                "law_name": "伪造法律",
                "article_number": "第一条",
                "semantic_chunk_id": "forged-article",
            },
            internal_metadata={
                "upload_task_id": "real-task",
                "upload_object_key": "tenants/t/uploads/real-task/payload.txt",
            },
        )

        assert result["status"] == "completed"
        metadata = fake_replace_document.calls[0]
        assert metadata["source_id"] == "stable-source"
        assert metadata["authority"] == "official"
        assert metadata["tenant_id"] == "00000000-0000-0000-0000-000000000001"
        assert metadata["filename"] == "actual.txt"
        assert metadata["document_id"] != "forged-document"
        assert metadata["upload_task_id"] == "real-task"
        assert metadata["upload_object_key"] == (
            "tenants/t/uploads/real-task/payload.txt"
        )
        assert "law_name" not in metadata
        assert "article_number" not in metadata
        assert "semantic_chunk_id" not in metadata

    asyncio.run(run())
