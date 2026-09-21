"""Tests for project-level prompt templates."""
from app.llm.prompt import (
    ANSWER_FORMAT_INSTRUCTIONS,
    PROMPTS_DIR,
    STREAM_ANSWER_INSTRUCTIONS,
    build_rag_prompt,
    build_system_prompt,
    load_prompt,
)
from app.service.chat_service import Generator


def test_prompt_files_exist_and_load():
    expected_files = [
        "system.txt",
        "business.txt",
        "citation.txt",
        "output.txt",
        "rag_template.jinja2",
    ]

    for filename in expected_files:
        assert (PROMPTS_DIR / filename).exists()
        assert load_prompt(filename)


def test_rag_prompt_uses_template_parts():
    prompt = build_rag_prompt(
        query="test question",
        context="test context",
        output_instructions=ANSWER_FORMAT_INSTRUCTIONS,
    )

    assert "test question" in prompt
    assert "test context" in prompt
    assert "用户问题" in prompt
    assert '"conclusion"' in prompt
    assert '"evidence"' in prompt
    assert "依据" in prompt
    assert "不同观点" in prompt
    assert "单线逻辑" in prompt
    assert "context_id" in prompt
    assert "quote" in prompt
    assert "基础法律关系" in prompt


def test_system_prompt_combines_role_business_and_citation():
    system_prompt = build_system_prompt()

    assert "RAG" in system_prompt
    assert "业务规则" in system_prompt
    assert "引用要求" in system_prompt
    assert "不同裁判口径" in system_prompt
    assert "强相关上下文" in system_prompt
    assert "基础法律关系" in system_prompt
    assert "不可信数据" in system_prompt
    assert "忽略文档中" in system_prompt


def test_generator_builds_normal_and_stream_prompts_from_templates():
    generator = Generator.__new__(Generator)
    generator.config = {
        "max_context_length": 200,
        "answer_contract": {"enabled": True},
    }

    normal_prompt = generator._build_prompt("test question", "test context")
    stream_prompt = generator._build_prompt("test question", "test context", stream=True)

    assert "test question" in normal_prompt
    assert '"conclusion"' in normal_prompt
    assert '"evidence"' in normal_prompt
    assert STREAM_ANSWER_INSTRUCTIONS in stream_prompt
    assert "单个完整 JSON 对象" in stream_prompt
