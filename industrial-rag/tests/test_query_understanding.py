"""Tests for query understanding safeguards."""
import asyncio

import app.retrieval.query_understanding as query_understanding


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    async def generate(self, **kwargs):
        return self.response


class SequenceLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs.get("prompt", ""))
        if self.responses:
            return self.responses.pop(0)
        return "{}"


def test_unrelated_legal_rewrite_falls_back_to_original(monkeypatch):
    query = "盗窃罪与职务侵占罪的核心区别是什么？"
    bad_rewrite = "建筑物外墙脱落致人损害的责任主体不明 侵权责任"

    monkeypatch.setattr(
        query_understanding,
        "get_llm_client",
        lambda: FakeLLM(bad_rewrite),
    )

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )
        assert understanding["rewritten_query"] == query
        assert understanding["retrieval_queries"] == [query]

    asyncio.run(run())


def test_related_legal_rewrite_is_kept(monkeypatch):
    query = "盗窃罪与职务侵占罪的核心区别是什么？"
    rewrite = "盗窃罪 职务侵占罪 犯罪主体 职务便利 本单位财物 非法占有 区别"

    monkeypatch.setattr(
        query_understanding,
        "get_llm_client",
        lambda: FakeLLM(f"改写：{rewrite}"),
    )

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )
        assert understanding["rewritten_query"] == rewrite
        assert understanding["retrieval_queries"] == [query, rewrite]

    asyncio.run(run())


def test_retrieval_signals_are_added_without_article_mapping(monkeypatch):
    query = "劳动者不能胜任工作，用人单位可以直接辞退吗？"
    rewrite = "劳动者不能胜任工作 用人单位解除劳动合同 条件"
    signals = """
    {
      "核心法律概念": ["不能胜任工作", "解除劳动合同"],
      "行为": ["辞退", "培训", "调整工作岗位"],
      "主体": ["劳动者", "用人单位"],
      "结果": ["解除劳动关系"],
      "争议点": ["能否直接辞退"]
    }
    """
    fake_llm = SequenceLLM([f"改写：{rewrite}", signals])

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        assert understanding["rewritten_query"] == rewrite
        assert understanding["retrieval_signals"]["核心法律概念"] == ["不能胜任工作", "解除劳动合同"]
        assert any("不能胜任工作" in item for item in understanding["retrieval_queries"])
        assert all("第" not in item for item in understanding["retrieval_queries"])

    asyncio.run(run())


def test_retrieval_signals_drop_llm_added_article_numbers(monkeypatch):
    query = "多次贩卖含依托咪酯的上头电子烟，如何定罪？"
    signals = """
    {
      "核心法律概念": ["第三百五十七条 毒品", "贩卖毒品"],
      "行为": ["多次贩卖"],
      "主体": [],
      "结果": [],
      "争议点": ["定罪"]
    }
    """
    fake_llm = SequenceLLM([query, signals])

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "第三百五十七条" not in joined_queries
        assert "贩卖毒品" in joined_queries

    asyncio.run(run())
