"""Tests for query understanding safeguards."""
import asyncio

import pytest

import app.retrieval.query_understanding as query_understanding
from app.retrieval.domain_signal_map import build_domain_signal_queries, load_domain_signal_rules
from app.retrieval.legal_concept_map import (
    load_legal_concept_article_mappings,
    match_legal_concept_articles,
)


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


def test_labeled_comparison_is_split_without_llm_ambiguity():
    query = (
        "情景一：甲向乙借款，乙要求担保。"
        "情景二：丙与丁签订劳动合同，合同期满后继续工作。"
        "请分别判断两个情景的处理结果，并说明理由。"
    )

    assert query_understanding._split_labeled_scenarios(query, 3) == [
        "甲向乙借款，乙要求担保",
        "丙与丁签订劳动合同，合同期满后继续工作",
    ]


def test_labeled_comparison_maps_each_scenario_without_cross_contamination(monkeypatch):
    query = (
        "情景一：张某发现某公司即将在短视频平台发布一段剪辑视频，内容涉及伪造其私生活丑闻，"
        "传播后必然造成其社会评价严重受损。张某向法院申请制止该行为。应如何处理？"
        "情景二：甲向乙借款十万元，双方约定该债权不得转让。后乙将该债权转让给丙，"
        "丙不知该约定。甲拒绝向丙还款。该债权转让是否有效？"
        "请分别判断两个情景的处理结果，并说明理由。"
    )
    fake_llm = SequenceLLM([query, "{}"])
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
        )

        mappings = {
            item["concept_id"]: {
                article["semantic_chunk_id"]
                for article in item["articles"]
            }
            for item in understanding["concept_article_mappings"]
        }
        assert set(mappings) == {
            "civil_personality_rights_preventive_injunction",
            "civil_claim_assignment_against_debtor",
        }
        assert mappings["civil_personality_rights_preventive_injunction"] == {
            "民法典_人格权_一般规定_997条"
        }
        assert mappings["civil_claim_assignment_against_debtor"] == {
            "民法典_合同_合同的变更和转让_545条"
        }

    asyncio.run(run())


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


def test_known_out_of_scope_query_short_circuits_external_model(monkeypatch):
    class FailLLM:
        async def generate(self, **kwargs):
            raise AssertionError("out-of-scope query must not call the LLM")

    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: FailLLM())

    async def run():
        result = await query_understanding.QueryUnderstanding().understand_query(
            query="发明专利优先审查条件具体如何规定？",
        )

        assert result["out_of_scope"] is True
        assert result["retrieval_queries"] == [result["resolved_query"]]
        assert result["retrieval_signals"] == {}

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


def test_rewrite_cannot_create_a_false_compound_query(monkeypatch):
    query = "劳动合同工资约定不明时怎样确定标准？"
    rewrite = "劳动合同工资约定不明以及怎样确定同工同酬标准"
    signals = '{"核心法律概念": ["劳动合同"], "行为": [], "主体": [], "结果": [], "争议点": []}'
    fake_llm = SequenceLLM([rewrite, signals])
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
        )

        assert understanding["is_decomposed"] is False
        assert understanding["subqueries"] == [query]
        assert understanding["rewritten_query"] == rewrite
        assert len(fake_llm.calls) == 2
        assert not any("多个可独立检索的子问题" in prompt for prompt in fake_llm.calls)

    asyncio.run(run())


def test_factual_respectively_word_does_not_trigger_decomposition(monkeypatch):
    query = (
        "甲向乙借款50万元，丙、丁分别与乙签订保证合同，均未约定保证份额。"
        "借款到期后甲无力偿还，乙要求丁承担全部50万元保证责任。丁应否承担？"
    )
    rewrite = "同一债务有两个以上保证人 未约定保证份额 保证责任"
    signals = '{"核心法律概念": ["共同保证"], "行为": [], "主体": [], "结果": [], "争议点": []}'
    fake_llm = SequenceLLM([rewrite, signals])
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
        )

        assert understanding["is_decomposed"] is False
        assert understanding["subqueries"] == [query]
        assert len(fake_llm.calls) == 2
        assert not any("多个可独立检索的子问题" in prompt for prompt in fake_llm.calls)

    asyncio.run(run())


def test_explicit_respectively_request_still_triggers_decomposition(monkeypatch):
    query = "甲和乙涉及不同争议，请分别判断甲的责任和乙的责任。"
    fake_llm = FakeLLM("甲应承担什么责任？\n乙应承担什么责任？")
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        result = await query_understanding.QueryUnderstanding().decompose_query(query)
        assert result == ["甲应承担什么责任？", "乙应承担什么责任？"]

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


def test_retrieval_signals_keep_common_crime_for_incitement_facts(monkeypatch):
    query = "甲教唆十五岁的乙盗窃，但乙没有实施。对甲应如何处理？"
    signals = """
    {
      "核心法律概念": ["教唆犯", "共同犯罪"],
      "行为": ["教唆未成年人盗窃"],
      "主体": [],
      "结果": ["被教唆人未实施犯罪"],
      "争议点": ["教唆犯的处罚"]
    }
    """
    fake_llm = SequenceLLM([query, signals])
    monkeypatch.setattr(query_understanding, "get_llm_client", lambda: fake_llm)

    async def run():
        understanding = await query_understanding.QueryUnderstanding().understand_query(
            query=query,
            enable_decomposition=False,
        )

        assert "共同犯罪" in understanding["retrieval_signals"]["核心法律概念"]

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


@pytest.mark.parametrize(
    ("concept_id", "article_id", "query"),
    [
        (
            "civil_step_parent_child_support",
            "民法典_婚姻家庭_家庭关系_父母子女关系和其他近亲属关系_1072条",
            "双方再婚后携子共同生活并长期负担学费，继子成年后是否应当赡养患病继父？",
        ),
        (
            "civil_agency_principal_death_unknown",
            "民法典_总则_代理_代理终止_173条",
            "房主委托代理人出售房屋后去世，代理人不知情仍签订买卖合同，合同效力如何？",
        ),
        (
            "civil_finance_lease_structure",
            "民法典_合同_融资租赁合同_735条",
            "融资公司按承租人指定购买机床并交付使用，承租人支付租金，制造缺陷责任如何处理？",
        ),
        (
            "civil_minor_own_income_capacity",
            "民法典_总则_自然人_民事权利能力和民事行为能力_18条",
            "17岁未成年人靠直播打赏月入八千并以此为主要生活来源，反悔买卖合同能否主张无效？",
        ),
        (
            "civil_joint_debt_partial_performance",
            "民法典_合同_合同的履行_520条",
            "连带债务人之一清偿十万元后，其他人对债权人的债务是否相应减少？",
        ),
        (
            "civil_limitation_voluntary_performance",
            "民法典_总则_诉讼时效_192条",
            "超过诉讼时效后债务人承诺偿清并先支付部分款项，随后反悔要求拿回应如何处理？",
        ),
        (
            "civil_indefinite_partnership_termination",
            "民法典_合同_合伙合同_976条",
            "合伙经营未约定合伙期限，一方通知终止并要求散伙是否有权解除？",
        ),
        (
            "criminal_omitted_offense_after_judgment",
            "刑法_总则_刑罚的具体运用_数罪并罚_70条",
            "罪犯服刑期间发现其在判决前还曾实施未处理的犯罪，检察机关起诉后如何处理？",
        ),
        (
            "criminal_probation_serious_violation",
            "刑法_总则_刑罚的具体运用_缓刑_77条",
            "缓刑期间未经批准离开居住地并多次违反禁令，情节严重时如何处理？",
        ),
        (
            "criminal_treaty_jurisdiction",
            "刑法_总则_刑法的任务基本原则和适用范围_9条",
            "外国人在公海外籍货轮策划袭击中国公民，依据中国加入的反恐公约如何处理？",
        ),
        (
            "criminal_smuggling_counterfeit_currency",
            "刑法_分则_破坏社会主义市场经济秩序罪_走私罪_151条",
            "从境外购得伪造欧元藏于汽车夹层走私入境并在海关查获，应如何定性？",
        ),
        (
            "criminal_inciting_violent_resistance",
            "刑法_分则_妨害社会管理秩序罪_扰乱公共秩序罪_278条",
            "煽动群众因国家政策砸毁镇政府办公楼并投掷石块，应如何定性？",
        ),
        (
            "criminal_death_penalty_age_seventy_five",
            "刑法_总则_刑罚_死刑_49条",
            "被告在75岁生日当天砍杀他人致其死亡，现处于审判阶段，应如何处理？",
        ),
        (
            "criminal_armed_rebellion_riot",
            "刑法_分则_危害国家安全罪_104条",
            "退役军官纠集多人组织持械冲击县政府，煽动打砸设施，应如何定性？",
        ),
        (
            "labor_collective_contract_union_enforcement",
            "劳动合同法_特别规定_集体合同_56条",
            "工会与公司签订集体合同后公司未履行并拖欠加班费，工会申请仲裁是否合法？",
        ),
        (
            "labor_wage_order_overdue_liability",
            "劳动合同法_法律责任_85条",
            "劳动行政部门责令公司限期补足低于最低工资的报酬，公司逾期仍未支付应负何责？",
        ),
        (
            "labor_written_contract_double_wage",
            "劳动合同法_劳动合同的订立_10条",
            "公司录用员工后未签书面协议，工作满两周要求补签，员工如何维权？",
        ),
        (
            "labor_unpaid_social_insurance_economic_compensation",
            "劳动合同法_劳动合同的解除和终止_38条",
            "公司未足额发放工资且拒绝为其缴纳社保，劳动者离职应如何处理？",
        ),
        (
            "civil_mandate_termination_incapacity",
            "民法典_合同_委托合同_934条",
            "甲委托乙代为出售名画，乙突发疾病成为植物人，家人能否继续履行合同？",
        ),
        (
            "criminal_specific_defective_product_fallback",
            "刑法_分则_破坏社会主义市场经济秩序罪_生产销售伪劣商品罪_149条",
            "生产不符合安全标准的食品但尚未造成严重危害，不构成特定犯罪且销售金额六万元，应如何处理？",
        ),
        (
            "criminal_sale_of_personal_information",
            "刑法_分则_侵犯公民人身权利民主权利罪_253条之一",
            "房产公司经理将客户姓名、电话和购房信息出售给装修商推销，客户不堪其扰，应如何定性？",
        ),
        (
            "criminal_state_owned_property_is_public_property",
            "刑法_总则_其他规定_91条",
            "将国有企业生产原料据为己有后辩称只是经营财产，该财产是否属于公共财产？",
        ),
        (
            "criminal_illegal_lending_of_firearm",
            "刑法_分则_危害公共安全罪_128条",
            "单位依规配发猎枪后，持枪人将猎枪借给朋友打猎，应如何定性？",
        ),
        (
            "labor_relationship_starts_on_actual_employment",
            "劳动合同法_劳动合同的订立_7条",
            "员工3月1日实际上班、4月才签合同，劳动关系应从哪一天起算？",
        ),
        (
            "labor_industry_or_regional_collective_contract",
            "劳动合同法_特别规定_集体合同_53条",
            "行业工会与多家餐饮企业代表订立覆盖全县的合同，未参与协商的餐厅拒绝执行，是否有约束力？",
        ),
        (
            "labor_contract_until_task_completion",
            "劳动合同法_劳动合同的订立_15条",
            "六个月工程提前完成，双方约定工程完工即合同终止，劳动者能否要求履行至原定期限？",
        ),
        (
            "labor_part_time_multiple_employers_oral_agreement",
            "劳动合同法_特别规定_非全日制用工_69条",
            "员工每天为甲公司保洁三小时，又与乙公司口头约定下午保洁，未签书面协议能否同时工作？",
        ),
        (
            "criminal_instigation_when_instigated_person_does_not_offend",
            "刑法_总则_犯罪_共同犯罪_29条",
            "甲教唆十五岁的乙盗窃，乙害怕而未实施，后甲单独盗窃得手，应如何处理？",
        ),
        (
            "labor_dispatch_temporary_auxiliary_substitute_positions",
            "劳动合同法_特别规定_劳务派遣_66条",
            "公司将核心生产岗位的长期工作全部改为劳务派遣，派遣工占40%并主张同等待遇，岗位性质如何认定？",
        ),
    ],
)
def test_high_specificity_concept_mappings_recall_exact_articles(
    concept_id: str,
    article_id: str,
    query: str,
):
    mapping = next(
        item
        for item in match_legal_concept_articles(query)
        if item["concept_id"] == concept_id
    )

    assert article_id in {
        article["semantic_chunk_id"] for article in mapping["articles"]
    }


def test_part_time_written_agreement_does_not_trigger_double_wage_mapping():
    mappings = match_legal_concept_articles(
        "劳动者分别在两家公司做非全日制保洁，其中一家公司称未签书面协议所以不存在用工关系。"
    )

    assert "labor_written_contract_double_wage" not in {
        mapping["concept_id"] for mapping in mappings
    }


@pytest.mark.parametrize(
    ("query", "concept_id"),
    [
        ("甲委托乙代为出售名画，乙尚未卖出。", "civil_mandate_termination_incapacity"),
        ("公司生产食品后销售金额六万元。", "criminal_specific_defective_product_fallback"),
        ("单位给保卫人员配发普通防护器材。", "criminal_illegal_lending_of_firearm"),
        ("员工每天在一家公司做保洁但未签书面协议。", "labor_part_time_multiple_employers_oral_agreement"),
    ],
)
def test_new_high_specificity_mappings_require_all_fact_groups(query, concept_id):
    matched_ids = {
        mapping["concept_id"] for mapping in match_legal_concept_articles(query)
    }

    assert concept_id not in matched_ids


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
