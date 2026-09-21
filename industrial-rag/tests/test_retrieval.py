"""Retrieval helper tests."""
import asyncio
import sys
import types

import pytest

from app.embedding import embedder
from app.retrieval import dense, factory, hybrid, reranker
from app.retrieval.domain_signal_map import is_known_out_of_scope, load_out_of_scope_signal_groups
from app.retrieval.legal_concept_map import match_legal_concept_articles
from app.retrieval.result_merge import merge_retrieval_results, result_score
from app.vectorstore import postgres_store
from app.vectorstore.postgres_store import _extract_chinese_keywords


def test_candidate_count_bounds():
    config = {
        "rerank_candidate_multiplier": 4,
        "rerank_min_candidates": 20,
        "rerank_max_candidates": 80,
    }

    assert dense._candidate_count(5, True, config) == 20
    assert dense._candidate_count(30, True, config) == 80
    assert hybrid._candidate_count(5, False, config) == 5


def test_retrieval_factory_respects_hybrid_switch(monkeypatch):
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: {"rag": {"retrieval": {"enable_hybrid": True}}},
    )
    assert isinstance(factory.build_retrieval_engine(), hybrid.HybridRetrievalEngine)

    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: {"rag": {"retrieval": {"enable_hybrid": False}}},
    )
    assert isinstance(factory.build_retrieval_engine(), dense.RetrievalEngine)


def test_dense_retriever_refuses_known_out_of_scope_before_embedding(monkeypatch):
    monkeypatch.setattr(
        dense,
        "encode_query",
        lambda query: (_ for _ in ()).throw(AssertionError("embedding must not run")),
    )

    result = asyncio.run(
        dense.RetrievalEngine().retrieve("增值税专用发票抵扣期限如何规定？")
    )

    assert result == []


def test_dense_deduplicates_by_document_and_chunk():
    docs = [
        {"id": "a", "score": 0.9, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "a_dup", "score": 0.8, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "b", "score": 0.7, "metadata": {"document_id": "doc1", "chunk_index": 1}},
        {"id": "c", "score": 0.6, "metadata": {"document_id": "doc2", "chunk_index": 0}},
    ]

    deduped = dense._deduplicate_results(docs, max_per_document=1)

    assert [doc["id"] for doc in deduped] == ["a", "c"]


def test_dense_retriever_preserves_explicit_legal_citation(monkeypatch):
    exact = {
        "id": "article-42",
        "content": "劳动合同法第四十二条",
        "score": 1.0,
        "metadata": {},
    }

    async def no_vector_results(**_kwargs):
        return []

    async def exact_lookup(pairs, partition=None):
        assert pairs == [("中华人民共和国劳动合同法", "第四十二条")]
        assert partition == "legal"
        return [exact]

    monkeypatch.setattr(dense, "_retrieval_config", lambda: {"enable_rerank": False})
    monkeypatch.setattr(dense, "encode_query", lambda _query: [0.0])
    monkeypatch.setattr(dense.storage_adapter, "search", no_vector_results)
    monkeypatch.setattr(
        dense.storage_adapter,
        "get_documents_by_citations",
        exact_lookup,
    )

    results = asyncio.run(
        dense.RetrievalEngine().retrieve(
            "劳动合同法第四十二条如何规定？",
            partition="legal",
            enable_rerank=False,
        )
    )

    assert results == [exact]


def test_reranker_closed_when_model_unavailable(monkeypatch):
    docs = [
        {"id": "a", "score": 0.9, "content": "first"},
        {"id": "b", "score": 0.8, "content": "second"},
    ]
    monkeypatch.setattr(reranker, "load_reranker", lambda: None)
    monkeypatch.setattr(
        reranker,
        "_reranker_config",
        lambda: {"enabled": True, "failure_mode": "closed"},
    )

    assert reranker.rerank_documents("query", docs, top_n=1) == []


def test_reranker_open_falls_back_when_model_unavailable(monkeypatch):
    docs = [
        {"id": "a", "score": 0.9, "content": "first"},
        {"id": "b", "score": 0.8, "content": "second"},
    ]
    monkeypatch.setattr(reranker, "load_reranker", lambda: None)
    monkeypatch.setattr(
        reranker,
        "_reranker_config",
        lambda: {"enabled": True, "failure_mode": "open"},
    )

    assert reranker.rerank_documents("query", docs, top_n=1) == [docs[0]]


def test_reranker_can_defer_threshold_for_merged_candidates(monkeypatch):
    class FakeReranker:
        def predict(self, pairs, batch_size):
            del pairs, batch_size
            return [0.4]

    monkeypatch.setattr(reranker, "load_reranker", lambda: FakeReranker())
    monkeypatch.setattr(
        reranker,
        "_reranker_config",
        lambda: {"enabled": True, "score_threshold": 0.5},
    )

    filtered = reranker.rerank_documents(
        "query", [{"id": "candidate", "content": "law"}], top_n=1
    )
    deferred = reranker.rerank_documents(
        "query",
        [{"id": "candidate", "content": "law"}],
        top_n=1,
        apply_threshold=False,
    )

    assert filtered == []
    assert [doc["id"] for doc in deferred] == ["candidate"]


def test_reranker_uses_child_passage_when_parent_context_is_present():
    doc = {
        "content": "large parent context",
        "child_content": "focused child passage",
        "metadata": {"parent_content": "large parent context"},
    }

    assert reranker._reranker_document_text(doc) == "focused child passage"


class _FakeSentenceTransformer:
    """Stand-in for SentenceTransformer that records truncation settings."""

    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device
        self.max_seq_length = 8192


def _load_with_embedding_config(monkeypatch, embed_config):
    """Load the embedder against a fake model and return the instance."""
    monkeypatch.setattr(embedder, "_embedding_model", None)
    monkeypatch.setattr(
        embedder,
        "get_settings",
        lambda: {"embedding": {"model_path": "/fake/bge-m3", "device": "cpu", **embed_config}},
    )
    monkeypatch.setattr(
        embedder,
        "resolve_torch_device",
        lambda device, **_kwargs: "cpu",
    )

    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    try:
        return embedder.load_embedding_model()
    finally:
        embedder._embedding_model = None


def test_embedding_max_length_is_applied_to_the_model(monkeypatch):
    """The setting feeds the corpus fingerprint, so it must govern truncation.

    If it stays inert, changing embedding.max_length invalidates the stored
    fingerprint and forces a corpus rebuild without altering a single vector.
    """
    model = _load_with_embedding_config(monkeypatch, {"max_length": 512})
    assert model.max_seq_length == 512


def test_embedding_max_length_above_model_limit_keeps_model_limit(monkeypatch):
    model = _load_with_embedding_config(monkeypatch, {"max_length": 99999})
    assert model.max_seq_length == 8192


def test_embedding_max_length_absent_leaves_model_default(monkeypatch):
    model = _load_with_embedding_config(monkeypatch, {})
    assert model.max_seq_length == 8192


def test_accelerator_fallback_is_rejected_for_strict_runtime(monkeypatch):
    unavailable_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", unavailable_torch)

    with pytest.raises(RuntimeError, match="accelerator cuda is unavailable"):
        embedder.resolve_torch_device("cuda", allow_cpu_fallback=False)


def test_accelerator_fallback_remains_available_for_laptop_runtime(monkeypatch):
    unavailable_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", unavailable_torch)

    assert embedder.resolve_torch_device("cuda", allow_cpu_fallback=True) == "cpu"


def test_embedding_max_length_rejects_non_positive_values(monkeypatch):
    with pytest.raises(ValueError, match="at least 1"):
        _load_with_embedding_config(monkeypatch, {"max_length": 0})


def test_result_score_uses_first_valid_rank_score():
    assert (
        result_score(
            {
                "score": "0.62",
                "rrf_score": 9.0,
                "metadata": {"rerank_prob": "not-a-number"},
            }
        )
        == 0.62
    )
    assert result_score({"metadata": {"rerank_prob": 0.0}, "score": 0.9}) == 0.0


def test_merge_prioritizes_controlled_article_ids():
    results = merge_retrieval_results(
        [
            [
                {"id": "unmapped-high", "score": 0.99, "metadata": {}},
                {
                    "id": "mapped-low",
                    "score": 0.2,
                    "metadata": {"semantic_chunk_id": "劳动合同法_劳动合同的订立_7条"},
                },
            ]
        ],
        top_k=1,
        queries=["mapped query"],
        priority_ids=["劳动合同法_劳动合同的订立_7条"],
    )

    assert [doc["id"] for doc in results] == ["mapped-low"]
    assert results[0]["mapped_article_priority"] == 1
    assert results[0]["source_relevance_score"] == 0.2
    assert results[0]["source_retrieval_rank"] == 2
    assert results[0]["source_query"] == "mapped query"
    assert results[0]["query_retrieval_ranks"] == {"mapped query": 2}


def test_merge_preserves_each_queries_independent_rank():
    results = merge_retrieval_results(
        [
            [
                {"id": "shared", "score": 0.3, "metadata": {}},
                {"id": "first-only", "score": 0.2, "metadata": {}},
            ],
            [
                {"id": "second-only", "score": 0.9, "metadata": {}},
                {"id": "shared", "score": 0.8, "metadata": {}},
            ],
        ],
        top_k=3,
        queries=["original", "rewrite"],
    )

    shared = next(doc for doc in results if doc["id"] == "shared")
    assert shared["query_retrieval_ranks"] == {"original": 1, "rewrite": 2}


def test_legal_keyword_extraction_uses_query_terms_without_article_mapping():
    keywords = _extract_chinese_keywords("多次贩卖含依托咪酯的上头电子烟，如何定罪？")

    assert "第三百四十七条" not in keywords
    assert "第三百五十七条" not in keywords
    assert any("依托咪酯" in keyword for keyword in keywords)
    assert any("电子烟" in keyword for keyword in keywords)


def test_legal_keyword_extraction_preserves_semantic_chunk_ids():
    keywords = _extract_chinese_keywords(
        "未缴社保 经济补偿 劳动合同法_劳动合同的解除和终止_46条"
    )

    assert "劳动合同法_劳动合同的解除和终止_46条" in keywords


def test_controlled_mappings_cover_comparison_and_coerced_accomplice():
    comparison = match_legal_concept_articles("盗窃罪与职务侵占罪的核心区别是什么？")
    comparison_ids = {
        article["semantic_chunk_id"]
        for mapping in comparison
        for article in mapping["articles"]
    }
    coerced = match_legal_concept_articles("受胁迫参加犯罪的胁从犯依法应如何处罚？")
    coerced_ids = {
        article["semantic_chunk_id"]
        for mapping in coerced
        for article in mapping["articles"]
    }

    assert "刑法_分则_侵犯财产罪_盗窃罪_264条" in comparison_ids
    assert "刑法_分则_侵犯财产罪_职务侵占罪_271条" in comparison_ids
    assert "刑法_总则_犯罪_共同犯罪_28条" in coerced_ids


def test_explicit_legal_citations_extracts_multiple_articles_and_longest_law_name():
    pairs = hybrid._explicit_legal_citations(
        "对比《中华人民共和国劳动合同法实施条例》第五条与第六条。"
    )
    assert pairs == [
        ("中华人民共和国劳动合同法实施条例", "第五条"),
        ("中华人民共和国劳动合同法实施条例", "第六条"),
    ]


def test_explicit_legal_citations_preserve_cross_law_pairing():
    assert hybrid._explicit_legal_citations(
        "比较《中华人民共和国民法典》第一条与《中华人民共和国刑法》第二条"
    ) == [
        ("中华人民共和国民法典", "第一条"),
        ("中华人民共和国刑法", "第二条"),
    ]


def test_explicit_legal_citations_normalize_common_law_aliases():
    assert hybrid._explicit_legal_citations(
        "比较民法典第一条、第二条与刑法第三条"
    ) == [
        ("中华人民共和国民法典", "第一条"),
        ("中华人民共和国民法典", "第二条"),
        ("中华人民共和国刑法", "第三条"),
    ]
    assert hybrid._explicit_legal_citations("劳动合同法实施条例第五条") == [
        ("中华人民共和国劳动合同法实施条例", "第五条")
    ]


def test_explicit_provisions_expand_beyond_fuzzy_top_k(monkeypatch):
    articles = ["第一条", "第二条", "第三条", "第四条", "第五条", "第六条"]

    async def fake_hybrid_search(**kwargs):
        return [{"id": "noise", "score": 0.9, "metadata": {}}]

    async def fake_exact_lookup(citation_pairs, partition=None):
        assert [article for _law, article in citation_pairs] == articles
        return [
            {
                "id": f"exact-{article}",
                "score": 1.0,
                "content": article,
                "metadata": {
                    "law_name": "中华人民共和国民法典",
                    "article_number": article,
                },
            }
            for article in articles
        ]

    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        hybrid.storage_adapter, "get_documents_by_citations", fake_exact_lookup
    )

    async def run():
        results = await hybrid.HybridRetrievalEngine().retrieve(
            "比较民法典第一条、第二条、第三条、第四条、第五条和第六条",
            top_k=5,
            enable_rerank=False,
            similarity_threshold=0.0,
        )
        assert [doc["id"] for doc in results] == [f"exact-{article}" for article in articles]

    asyncio.run(run())


def test_internal_recall_query_cannot_trigger_exact_citation_lookup(monkeypatch):
    exact_calls = []

    async def fake_hybrid_search(**kwargs):
        return [{"id": "fuzzy", "score": 0.9, "metadata": {}}]

    async def fake_exact_lookup(citation_pairs, partition=None):
        exact_calls.append((citation_pairs, partition))
        return [{"id": "exact", "score": 1.0, "metadata": {}}]

    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        hybrid.storage_adapter, "get_documents_by_citations", fake_exact_lookup
    )

    async def run():
        results = await hybrid.HybridRetrievalEngine().retrieve(
            "中华人民共和国刑法 第二十八条 胁从犯",
            enable_rerank=False,
            similarity_threshold=0.0,
            enable_exact_citations=False,
        )
        assert [doc["id"] for doc in results] == ["fuzzy"]

    asyncio.run(run())
    assert exact_calls == []


def test_out_of_scope_signals_distinguish_procedure_from_criminal_provision():
    assert len(load_out_of_scope_signal_groups()) >= 2
    assert is_known_out_of_scope("增值税专用发票抵扣期限如何规定") is True
    assert is_known_out_of_scope("出口退税备案材料有哪些") is True
    assert is_known_out_of_scope("无线电频率许可如何申请") is True
    assert is_known_out_of_scope("证券内幕交易信息披露具体如何规定") is True
    assert is_known_out_of_scope("不动产登记收费标准具体如何规定") is True
    assert is_known_out_of_scope("虚开增值税专用发票如何定罪") is False
    assert is_known_out_of_scope("无线电设备损坏应如何承担侵权责任") is False


def test_keyword_search_applies_partition_to_all_or_terms(monkeypatch):
    captured = {}

    class FakeCursor:
        def execute(self, sql, params):
            captured["sql"] = " ".join(sql.split())
            captured["params"] = params

        def fetchall(self):
            return []

        def close(self):
            pass

    class FakeConnection:
        def rollback(self):
            captured["rolled_back"] = True

        def cursor(self, *args, **kwargs):
            return FakeCursor()

    class FakePool:
        def putconn(self, conn, close=False):
            captured["returned_connection"] = conn
            captured["connection_discarded"] = close

    monkeypatch.setattr(postgres_store, "_connection", lambda: FakeConnection())
    monkeypatch.setattr(postgres_store, "_pool", FakePool())
    monkeypatch.setattr(postgres_store, "_extract_chinese_keywords", lambda query: ["社保", "补偿"])

    assert postgres_store._keyword_search_sync("社保补偿", top_k=5, partition="labor") == []
    assert "WHERE tenant_id = %s::uuid AND (" in captured["sql"]
    assert ") AND partition = %s" in captured["sql"]
    assert captured["params"][2] == "00000000-0000-0000-0000-000000000001"
    assert captured["params"][-2:] == ["labor", 5]
    assert captured["rolled_back"] is True
    assert captured["connection_discarded"] is False


def test_failed_rollback_discards_connection(monkeypatch):
    captured = {}

    class BrokenConnection:
        def rollback(self):
            raise RuntimeError("connection lost")

    class FakePool:
        def putconn(self, conn, close=False):
            captured["connection"] = conn
            captured["discarded"] = close

    connection = BrokenConnection()
    monkeypatch.setattr(postgres_store, "_pool", FakePool())

    postgres_store._return_connection(connection)

    assert captured == {"connection": connection, "discarded": True}


def test_postgres_initialization_failure_disposes_runtime(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False

        def closeall(self):
            self.closed = True

    fake_pool = FakePool()
    monkeypatch.setattr(postgres_store, "_pool", None)
    monkeypatch.setattr(postgres_store, "_executor", None)
    monkeypatch.setattr(
        postgres_store.psycopg2.pool,
        "ThreadedConnectionPool",
        lambda *args, **kwargs: fake_pool,
    )

    async def fail_schema(*args, **kwargs):
        raise PermissionError("schema denied")

    monkeypatch.setattr(postgres_store, "_execute_sync", fail_schema)

    async def run():
        with pytest.raises(PermissionError, match="schema denied"):
            await postgres_store.init_postgres_store()

    asyncio.run(run())
    assert fake_pool.closed is True
    assert postgres_store._pool is None
    assert postgres_store._executor is None


def test_schema_metadata_rejects_incompatible_populated_corpus(monkeypatch):
    expected = {
        "schema_version": "1",
        "embedding_dimension": "1024",
        "embedding_fingerprint": "embedding-current",
        "chunking_fingerprint": "chunking-current",
    }
    monkeypatch.setattr(postgres_store, "_runtime_schema_metadata", lambda: expected)

    with pytest.raises(RuntimeError, match="embedding_fingerprint"):
        postgres_store._validate_schema_metadata(
            {**expected, "embedding_fingerprint": "embedding-old"},
            document_count=10,
        )


def test_schema_metadata_allows_first_fingerprint_backfill(monkeypatch):
    expected = {
        "schema_version": "1",
        "embedding_dimension": "1024",
        "embedding_fingerprint": "embedding-current",
        "chunking_fingerprint": "chunking-current",
    }
    monkeypatch.setattr(postgres_store, "_runtime_schema_metadata", lambda: expected)

    postgres_store._validate_schema_metadata(
        {"schema_version": "1", "embedding_dimension": "1024"},
        document_count=10,
    )


def test_embedding_fingerprint_changes_with_inference_runtime(monkeypatch):
    versions = {
        "torch": "2.13.0+cpu",
        "transformers": "5.16.1",
        "sentence-transformers": "6.0.0",
        "tokenizers": "0.23.1",
    }
    monkeypatch.setattr(postgres_store, "_inference_runtime_versions", lambda: versions)
    current = postgres_store._runtime_schema_metadata()["embedding_fingerprint"]

    monkeypatch.setattr(
        postgres_store,
        "_inference_runtime_versions",
        lambda: {**versions, "transformers": "5.16.2"},
    )
    upgraded = postgres_store._runtime_schema_metadata()["embedding_fingerprint"]

    assert upgraded != current


def test_retrievers_default_threshold_matches_config_default(monkeypatch):
    docs = [
        {"id": "low", "score": 0.04, "metadata": {"document_id": "doc1", "chunk_index": 0}},
        {"id": "kept", "score": 0.05, "metadata": {"document_id": "doc1", "chunk_index": 1}},
    ]

    async def fake_search(**kwargs):
        return docs

    async def fake_hybrid_search(**kwargs):
        return docs

    monkeypatch.setattr(dense, "_retrieval_config", lambda: {})
    monkeypatch.setattr(hybrid, "_retrieval_config", lambda: {})
    monkeypatch.setattr(dense, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])
    monkeypatch.setattr(dense.storage_adapter, "search", fake_search)
    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)

    async def run():
        dense_results = await dense.RetrievalEngine().retrieve("query", enable_rerank=False)
        hybrid_results = await hybrid.HybridRetrievalEngine().retrieve("query", enable_rerank=False)

        assert [doc["id"] for doc in dense_results] == ["kept"]
        assert [doc["id"] for doc in hybrid_results] == ["kept"]

    asyncio.run(run())


def test_hybrid_engine_passes_runtime_fusion_switches(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        hybrid,
        "_retrieval_config",
        lambda: {
            "enable_rrf": False,
            "enable_dynamic_topk": False,
            "similarity_threshold": 0.0,
            "enable_rerank": False,
        },
    )
    monkeypatch.setattr(hybrid, "encode_query", lambda query: [0.0])

    async def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(hybrid.storage_adapter, "hybrid_search", fake_hybrid_search)

    asyncio.run(hybrid.HybridRetrievalEngine().retrieve("普通法律问题", enable_rerank=False))

    assert captured["enable_rrf"] is False
    assert captured["enable_dynamic_topk"] is False
