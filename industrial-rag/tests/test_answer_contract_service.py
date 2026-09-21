import asyncio
import json

import pytest

from app.llm.answer_contract import AnswerContractError
from app.service import chat_service
from app.service.chat_service import Generator
from app.service.enhanced_query_service import EnhancedQueryService


def _generator(outputs: list[str]) -> tuple[Generator, list[str]]:
    calls: list[str] = []

    class FakeLLM:
        async def generate(self, **kwargs):
            calls.append("generate")
            return outputs.pop(0)

        async def generate_stream(self, **kwargs):
            calls.append("stream")
            yield outputs.pop(0)

    generator = Generator.__new__(Generator)
    generator.config = {
        "answer_contract": {"enabled": True, "max_retries": 1},
        "max_context_per_doc": 1200,
        "max_context_length": 4000,
    }
    generator.llm = FakeLLM()
    return generator, calls


def _valid_answer() -> str:
    return json.dumps(
        {
            "conclusion": "用人单位自用工之日起建立劳动关系。",
            "evidence": [
                {
                    "claim": "用人单位自用工之日起建立劳动关系",
                    "context_id": "ctx-1",
                    "quote": "第七条 用人单位自用工之日起建立劳动关系。",
                }
            ],
            "insufficient_context": False,
        },
        ensure_ascii=False,
    )


def _docs() -> list[dict]:
    return [
        {
            "id": "law-7",
            "content": "第七条 用人单位自用工之日起建立劳动关系。",
            "metadata": {"filename": "labor-law.txt"},
        }
    ]


def test_generation_retries_once_then_formats_server_verified_evidence():
    async def run() -> None:
        generator, calls = _generator(["未经验证的草稿", _valid_answer()])
        answer = await generator.generate("劳动关系何时建立？", _docs(), use_cache=False)

        assert calls == ["generate", "generate"]
        assert "未经验证的草稿" not in answer
        assert "labor-law.txt" in answer
        assert "核验原文：第七条 用人单位自用工之日起建立劳动关系。" in answer

    asyncio.run(run())


def test_structured_retry_is_deterministic_and_targets_missing_claim_evidence():
    async def run() -> None:
        calls: list[dict] = []
        invalid = json.dumps(
            {
                "conclusion": "用人单位自用工之日起建立劳动关系。另有未支持结论。",
                "evidence": [
                    {
                        "claim": "用人单位自用工之日起建立劳动关系",
                        "context_id": "ctx-1",
                        "quote": "第七条 用人单位自用工之日起建立劳动关系。",
                    }
                ],
                "insufficient_context": False,
            },
            ensure_ascii=False,
        )
        outputs = [invalid, _valid_answer()]

        class RecordingLLM:
            async def generate(self, **kwargs):
                calls.append(kwargs)
                return outputs.pop(0)

        generator = Generator.__new__(Generator)
        generator.config = {
            "answer_contract": {"enabled": True, "max_retries": 1},
            "temperature": 0.9,
            "max_context_per_doc": 1200,
            "max_context_length": 4000,
        }
        generator.llm = RecordingLLM()

        answer = await generator.generate("劳动关系何时建立？", _docs(), use_cache=False)

        assert len(calls) == 2
        assert all(call["temperature"] == 0 for call in calls)
        assert all(call["structured_output"] is True for call in calls)
        assert "仅包含 evidence 已直接覆盖的一至三个短分句" in calls[1]["prompt"]
        assert "labor-law.txt" in answer

    asyncio.run(run())


def test_final_retry_recovers_only_the_grounded_claim_and_caches_canonical_json():
    async def run() -> None:
        invalid = json.dumps(
            {
                "conclusion": "用人单位自用工之日起建立劳动关系。另有未支持结论。",
                "evidence": [
                    {
                        "claim": "用人单位自用工之日起建立劳动关系",
                        "context_id": "ctx-1",
                        "quote": "第七条 用人单位自用工之日起建立劳动关系。",
                        "source": "model-controlled-source",
                    }
                ],
                "insufficient_context": False,
                "sources": ["model-controlled-source"],
            },
            ensure_ascii=False,
        )
        generator, calls = _generator(["not-json", invalid])

        validated, canonical = await generator._generate_contract_answer(
            "劳动关系何时建立？",
            _docs(),
        )

        assert calls == ["generate", "generate"]
        assert validated is not None
        assert validated.conclusion == "用人单位自用工之日起建立劳动关系。"
        assert canonical is not None
        assert "未支持结论" not in canonical
        assert "model-controlled-source" not in canonical

    asyncio.run(run())


def test_aggregate_recovery_runs_only_after_strict_attempts_are_exhausted():
    async def run() -> None:
        generator, calls = _generator([])
        service = EnhancedQueryService(
            query_understanding=object(),  # type: ignore[arg-type]
            retrieval_engine=object(),  # type: ignore[arg-type]
            generator=generator,
        )
        invalid = json.dumps(
            {
                "conclusion": "用人单位自用工之日起建立劳动关系。另有未支持结论。",
                "evidence": [
                    {
                        "claim": "用人单位自用工之日起建立劳动关系",
                        "context_id": "ctx-1",
                        "quote": "第七条 用人单位自用工之日起建立劳动关系。",
                    }
                ],
                "insufficient_context": False,
            },
            ensure_ascii=False,
        )

        answer = await service._aggregate_contract_answer(
            "劳动关系何时建立？",
            [],
            "unused",
            _docs(),
            initial_raw_answer=invalid,
            max_attempts=1,
        )

        assert calls == []
        assert "用人单位自用工之日起建立劳动关系" in answer
        assert "未支持结论" not in answer
        assert "labor-law.txt" in answer

    asyncio.run(run())


def test_streaming_buffers_invalid_output_and_fails_closed_after_one_retry():
    async def run() -> None:
        generator, calls = _generator(["draft", "still not JSON"])
        chunks = [chunk async for chunk in generator.generate_stream("问题", _docs())]
        answer = "".join(chunks)

        assert calls == ["stream", "stream"]
        assert "draft" not in answer
        assert "当前知识库没有找到足够相关的信息" in answer

    asyncio.run(run())


def test_validation_map_contains_only_the_final_globally_bounded_context():
    generator, _calls = _generator([])
    generator.config["max_context_length"] = 1300
    docs = [
        {
            "id": "long-law",
            "content": "第七条 劳动合同依法订立。" + ("普通文本" * 500) + "应处有期徒刑。",
            "metadata": {"filename": "long-law.txt"},
        }
    ]

    rendered = generator._build_context(docs)
    context_map = generator._context_map(docs)
    assert len(rendered) <= 1300
    assert context_map["ctx-1"] in rendered
    assert "有期徒刑" not in context_map["ctx-1"]

    raw = json.dumps(
        {
            "conclusion": "应处有期徒刑。",
            "evidence": [
                {
                    "claim": "应处有期徒刑",
                    "context_id": "ctx-1",
                    "quote": "应处有期徒刑。",
                }
            ],
            "insufficient_context": False,
        },
        ensure_ascii=False,
    )
    with pytest.raises(AnswerContractError, match="exact context substring"):
        generator.parse_contract_answer(raw, docs)


def test_semantic_cache_fingerprint_hashes_the_actual_child_prompt_content(monkeypatch):
    generator, _calls = _generator([])
    monkeypatch.setattr(chat_service, "get_settings", lambda: {"llm": {"text": {}}})
    generator._get_system_prompt = lambda: "system"
    base = {
        "id": "same-id",
        "content": "shared parent",
        "metadata": {"filename": "law.txt"},
    }

    first = generator._cache_context([{**base, "child_content": "child A"}])
    second = generator._cache_context([{**base, "child_content": "child B"}])

    assert first["rendered_context_sha256"] != second["rendered_context_sha256"]
    assert first["documents"][0]["content_sha256"] != second["documents"][0]["content_sha256"]


def test_disabled_answer_contract_uses_plain_text_output_protocol():
    generator, _calls = _generator([])
    generator.config["answer_contract"] = {"enabled": False}

    prompt = generator._build_prompt("问题", generator._build_context(_docs()))

    assert "直接输出最终答案正文" in prompt
    assert '"conclusion"' not in prompt


def test_explicit_citation_coverage_fails_closed_before_generation():
    async def run() -> None:
        generator, calls = _generator([])
        generator.config["max_context_length"] = 110
        docs = [
            {
                "id": "civil-code-1",
                "content": "第一条 为了保护民事主体的合法权益，制定本法。",
                "metadata": {
                    "filename": "民法典.txt",
                    "law_name": "中华人民共和国民法典",
                    "article_number": "第一条",
                },
            },
            {
                "id": "civil-code-2",
                "content": "第二条 民法调整平等主体之间的人身关系和财产关系。",
                "metadata": {
                    "filename": "民法典.txt",
                    "law_name": "中华人民共和国民法典",
                    "article_number": "第二条",
                },
            },
        ]

        answer = await generator.generate(
            "比较民法典第一条与第二条",
            docs,
            use_cache=False,
        )

        assert calls == []
        assert "《中华人民共和国民法典》第二条" in answer
        assert "未被完整覆盖" in answer

    asyncio.run(run())
