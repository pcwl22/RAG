"""Tests for query understanding safeguards."""
import asyncio

import pytest

import app.retrieval.query_understanding as query_understanding
from app.retrieval.domain_signal_map import build_domain_signal_queries, load_domain_signal_rules
from app.retrieval.legal_concept_map import load_legal_concept_article_mappings


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
        assert understanding["retrieval_queries"][0] == query
        assert bad_rewrite not in understanding["retrieval_queries"]
        assert any("职务侵占罪" in item for item in understanding["retrieval_queries"])

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
        assert understanding["retrieval_queries"][:2] == [query, rewrite]
        assert any("职务侵占罪" in item for item in understanding["retrieval_queries"])

    asyncio.run(run())


def test_retrieval_signals_and_controlled_article_mapping_are_added(monkeypatch):
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
        assert understanding["concept_article_mappings"]
        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "劳动合同法_劳动合同的解除和终止_40条" in joined_queries
        assert "第四十条" in joined_queries

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


def test_labor_dismissal_domain_signals_do_not_add_article_numbers():
    query = (
        "\u516c\u53f8\u8ba9\u5458\u5de5\u7b7e\u8ba2\u7a7a\u767d"
        "\u52b3\u52a8\u5408\u540c\u540e\u53c8\u4ee5\u6b64\u8f9e\u9000"
    )
    domain_queries = build_domain_signal_queries(query, "", "")

    assert domain_queries
    joined = "\n".join(domain_queries)
    assert "\u7528\u4eba\u5355\u4f4d\u5355\u65b9\u89e3\u9664\u52b3\u52a8\u5408\u540c" in joined
    assert "\u8fdd\u53cd\u672c\u6cd5\u89c4\u5b9a\u89e3\u9664\u6216\u8005\u7ec8\u6b62\u52b3\u52a8\u5408\u540c" in joined
    assert "\u7b2c\u4e09\u5341\u4e5d\u6761" not in joined
    assert "\u7b2c\u56db\u5341\u6761" not in joined


def test_domain_signal_rule_loader_validates_nested_shape(tmp_path):
    rule_path = tmp_path / "bad_domain_signal_rules.json"
    rule_path.write_text(
        '[{"rule_id":"bad","match_all":[[]],"queries":["recall"]}]',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"match_all\[1\].*non-empty list"):
        load_domain_signal_rules(str(rule_path))


def test_legal_concept_loader_validates_article_shape(tmp_path):
    mapping_path = tmp_path / "bad_legal_concept_mappings.json"
    mapping_path.write_text(
        """
        [{
          "concept_id": "bad",
          "concept": "bad concept",
          "match_all": [["bad"]],
          "articles": [{"law": "测试法", "article": "第一条"}],
          "recall_terms": ["bad recall"]
        }]
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic_chunk_id"):
        load_legal_concept_article_mappings(str(mapping_path))


def test_social_insurance_concept_maps_to_labor_contract_articles(monkeypatch):
    query = "劳动者承诺自愿放弃社保后，还能以未缴社保为由要求经济补偿吗？"
    rewrite = "劳动者自愿放弃社保后主张未缴社保经济补偿的效力认定"
    signals = """
    {
      "核心法律概念": ["自愿放弃社保", "未缴社保", "经济补偿"],
      "行为": ["承诺放弃", "要求补偿"],
      "主体": ["劳动者"],
      "结果": ["能否要求经济补偿"],
      "争议点": ["自愿放弃社保的效力", "未缴社保与经济补偿的关系"]
    }
    """
    fake_llm = SequenceLLM([f"改写：{rewrite}", signals])

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        mappings = understanding["concept_article_mappings"]
        assert mappings[0]["concept_id"] == "labor_unpaid_social_insurance_economic_compensation"
        articles = {article["article"] for article in mappings[0]["articles"]}
        assert {"第三十八条", "第四十六条"}.issubset(articles)
        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "未依法为劳动者缴纳社会保险费" in joined_queries
        assert "劳动合同法_劳动合同的解除和终止_46条" in joined_queries

    asyncio.run(run())


def test_worker_refusal_maps_to_implementation_regulation_articles(monkeypatch):
    query = "劳动者故意拒绝订立书面劳动合同，用人单位还要支付二倍工资吗？"
    fake_llm = SequenceLLM([query, "{}"])
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        mapping = next(
            item
            for item in understanding["concept_article_mappings"]
            if item["concept_id"] == "labor_written_contract_double_wage"
        )
        article_ids = {article["semantic_chunk_id"] for article in mapping["articles"]}
        assert "劳动合同法实施条例_劳动合同的订立_5条" in article_ids
        assert "劳动合同法实施条例_劳动合同的订立_6条" in article_ids
        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "书面通知劳动者终止劳动关系" in joined_queries
        assert "支付两倍工资" in joined_queries

    asyncio.run(run())


def test_affiliated_enterprise_mixed_employment_maps_to_labor_relationship_articles(monkeypatch):
    query = "关联企业混同用工，劳动者该向谁主张权利？"
    rewrite = "关联企业混同用工情形下劳动关系的认定与用人单位责任承担"
    signals = """
    {
      "核心法律概念": ["关联企业", "混同用工", "劳动关系认定"],
      "行为": ["混同用工"],
      "主体": ["劳动者", "关联企业", "用人单位"],
      "结果": ["主张权利"],
      "争议点": ["责任主体确定"]
    }
    """
    fake_llm = SequenceLLM([f"改写：{rewrite}", signals])

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        mappings = understanding["concept_article_mappings"]
        assert mappings[0]["concept_id"] == "labor_affiliated_enterprise_mixed_employment"
        articles = {article["article"] for article in mappings[0]["articles"]}
        assert {"第二条", "第七条", "第十条"}.issubset(articles)
        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "劳动合同法_总则_2条" in joined_queries
        assert "劳动合同法_劳动合同的订立_7条" in joined_queries
        assert "用人单位自用工之日起即与劳动者建立劳动关系" in joined_queries

    asyncio.run(run())


def test_receiving_selling_criminal_proceeds_maps_to_criminal_law_312(monkeypatch):
    query = "假借废品回收便利长期收赃销赃，应如何惩处？"
    rewrite = "掩饰隐瞒犯罪所得罪 收赃销赃 废品回收 长期行为 刑事处罚"
    signals = """
    {
      "核心法律概念": ["收赃", "销赃", "掩饰、隐瞒犯罪所得"],
      "行为": ["假借废品回收便利", "长期收购", "代为销售"],
      "主体": ["废品回收经营者"],
      "结果": ["刑事处罚"],
      "争议点": ["长期收赃销赃如何惩处"]
    }
    """
    fake_llm = SequenceLLM([f"改写：{rewrite}", signals])

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        signals = understanding["retrieval_signals"]["核心法律概念"]
        assert "共同犯罪" not in signals
        assert "连续犯" not in signals
        mappings = understanding["concept_article_mappings"]
        assert mappings[0]["concept_id"] == "criminal_receiving_selling_criminal_proceeds"
        articles = {article["article"] for article in mappings[0]["articles"]}
        assert "第三百一十二条" in articles
        joined_queries = "\n".join(understanding["retrieval_queries"])
        assert "刑法_分则_妨害社会管理秩序罪_妨害司法罪_312条" in joined_queries
        assert "掩饰、隐瞒犯罪所得、犯罪所得收益罪" in joined_queries

    asyncio.run(run())
