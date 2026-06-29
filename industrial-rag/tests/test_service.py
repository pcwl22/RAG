"""Service orchestration tests."""
import asyncio

import app.service.ingest_service as ingest_service
from app.service.chat_service import Generator


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

        formatted = generator._format_answer("fake answer body", docs)
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

        formatted = generator._format_answer(decline, docs)
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
    formatted = generator._format_answer(answer, docs, "多次贩卖含依托咪酯的上头电子烟，如何定罪？")

    assert "第三百五十七条" in formatted
    assert "第五十二条" not in formatted


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
                "content": "parent content " * 200,
                "child_content": "第二百七十一条 利用职务上的便利，将本单位财物非法占为己有。",
                "metadata": {"filename": "criminal-law.docx"},
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

        async def fake_add_documents(ids, embeddings, documents, metadatas=None, partition="general"):
            fake_add_documents.calls = {
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
        monkeypatch.setattr(ingest_service, "_storage_backend", lambda: fake_add_documents)

        result = await ingest_service.process_document("unused.docx", "中华人民共和国测试法_20240101.docx")

        metadatas = fake_add_documents.calls["metadatas"]
        assert result["status"] == "completed"
        assert result["total_chunks"] == 2
        assert fake_add_documents.calls["documents"][0].startswith("第一条")
        assert metadatas[0]["document_type"] == "legal_article"
        assert metadatas[0]["law_name"] == "中华人民共和国测试法"
        assert metadatas[0]["law_chapter"] == "第一章 基本规定"
        assert metadatas[0]["article_number"] == "第一条"
        assert metadatas[0]["article_text"] == fake_add_documents.calls["documents"][0]

    asyncio.run(run())


def test_process_document_orchestrates_parser_embedding_and_storage(monkeypatch):
    async def run():
        sample_text = ("Article 1. Module smoke test content.\n\n" * 20).strip()

        async def fake_add_documents(ids, embeddings, documents, metadatas=None, partition="general"):
            fake_add_documents.calls = {
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
        monkeypatch.setattr(ingest_service, "_storage_backend", lambda: fake_add_documents)

        result = await ingest_service.process_document("unused.txt", "demo.txt")

        assert result["status"] == "completed"
        assert result["total_chunks"] == 2
        assert len(fake_add_documents.calls["ids"]) == 2
        assert fake_add_documents.calls["metadatas"][0]["filename"] == "demo.txt"

    asyncio.run(run())
