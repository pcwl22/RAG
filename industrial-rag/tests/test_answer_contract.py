import json

import pytest

from app.llm.answer_contract import (
    AnswerContractError,
    recover_grounded_answer,
    serialize_validated_answer,
    validate_answer,
)


def test_structured_answer_accepts_only_exact_retrieved_quote():
    contexts = {"ctx-1": "第七条 用人单位自用工之日起建立劳动关系。"}

    answer = validate_answer(
        json.dumps(
            {
                "conclusion": "用人单位自用工之日起建立劳动关系。",
                "evidence": [
                    {
                        "claim": "用人单位自用工之日起建立劳动关系",
                        "context_id": "ctx-1",
                        "quote": contexts["ctx-1"],
                    }
                ],
                "insufficient_context": False,
            },
            ensure_ascii=False,
        ),
        contexts,
    )

    assert answer.conclusion.startswith("用人单位")
    assert answer.evidence[0].context_id == "ctx-1"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "conclusion": "有依据的结论",
            "evidence": [{"claim": "有依据的结论", "context_id": "ctx-9", "quote": "原文"}],
            "insufficient_context": False,
        },
        {
            "conclusion": "有依据的结论",
            "evidence": [
                {"claim": "有依据的结论", "context_id": "ctx-1", "quote": "改写后的原文"}
            ],
            "insufficient_context": False,
        },
        {
            "conclusion": "依据不存在的第三百九十九条",
            "evidence": [
                {
                    "claim": "依据不存在的第三百九十九条",
                    "context_id": "ctx-1",
                    "quote": "原文",
                }
            ],
            "insufficient_context": False,
        },
        {
            "conclusion": "有依据的结论",
            "evidence": [{"claim": "有依据的结论", "context_id": "ctx-1", "quote": "原文"}],
            "insufficient_context": False,
            "sources": ["model-generated-source"],
        },
    ],
)
def test_structured_answer_rejects_untrusted_citations(payload):
    with pytest.raises(AnswerContractError):
        validate_answer(json.dumps(payload, ensure_ascii=False), {"ctx-1": "原文"})


def test_structured_answer_allows_deterministic_insufficient_context():
    answer = validate_answer(
        json.dumps(
            {
                "conclusion": "当前资料不足，无法可靠回答。",
                "evidence": [],
                "insufficient_context": True,
            },
            ensure_ascii=False,
        ),
        {"ctx-1": "无关资料"},
    )

    assert answer.insufficient_context is True
    assert answer.evidence == ()


@pytest.mark.parametrize("conclusion", ["应判处死刑。", "应赔偿100万元。"])
def test_structured_answer_rejects_unsupported_outcome_or_numeric_claim(conclusion):
    payload = {
        "conclusion": conclusion,
        "evidence": [
            {
                "claim": conclusion.rstrip("。"),
                "context_id": "ctx-1",
                "quote": "第七条 用人单位建立劳动关系。",
            }
        ],
        "insufficient_context": False,
    }

    with pytest.raises(AnswerContractError):
        validate_answer(json.dumps(payload, ensure_ascii=False), {"ctx-1": payload["evidence"][0]["quote"]})


def test_each_claim_is_checked_only_against_its_cited_evidence():
    contexts = {
        "ctx-1": "第七条 劳动合同依法订立。",
        "ctx-2": "犯罪行为应处有期徒刑三年。",
    }
    payload = {
        "conclusion": "该行为应处有期徒刑。",
        "evidence": [
            {
                "claim": "该行为应处有期徒刑",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
            }
        ],
        "insufficient_context": False,
    }

    with pytest.raises(AnswerContractError, match="unsupported legal consequence"):
        validate_answer(json.dumps(payload, ensure_ascii=False), contexts)


def test_every_conclusion_clause_requires_an_explicit_claim_binding():
    contexts = {"ctx-1": "合同依法成立。"}
    payload = {
        "conclusion": "合同依法成立。还应赔偿损失。",
        "evidence": [
            {
                "claim": "合同依法成立",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
            }
        ],
        "insufficient_context": False,
    }

    with pytest.raises(AnswerContractError, match="every conclusion claim"):
        validate_answer(json.dumps(payload, ensure_ascii=False), contexts)


def test_single_shared_legal_phrase_cannot_ground_an_unrelated_claim():
    contexts = {"ctx-1": "用人单位应当按时足额支付劳动报酬。"}
    payload = {
        "conclusion": "用人单位可以随时解除劳动合同。",
        "evidence": [
            {
                "claim": "用人单位可以随时解除劳动合同",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
            }
        ],
        "insufficient_context": False,
    }

    with pytest.raises(AnswerContractError, match="claim is not grounded"):
        validate_answer(json.dumps(payload, ensure_ascii=False), contexts)


def test_close_paraphrase_with_substantial_overlap_remains_valid():
    contexts = {"ctx-1": "用人单位自用工之日起建立劳动关系。"}
    payload = {
        "conclusion": "劳动关系自实际用工之日起成立。",
        "evidence": [
            {
                "claim": "劳动关系自实际用工之日起成立",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
            }
        ],
        "insufficient_context": False,
    }

    answer = validate_answer(json.dumps(payload, ensure_ascii=False), contexts)
    assert answer.insufficient_context is False


def test_recovery_keeps_only_independently_grounded_allowlisted_evidence():
    contexts = {
        "ctx-1": "第七条 用人单位自用工之日起建立劳动关系。",
        "ctx-2": "用人单位应当按时足额支付劳动报酬。",
    }
    payload = {
        "conclusion": "用人单位自用工之日起建立劳动关系。用人单位可以随时解除劳动合同。",
        "evidence": [
            {
                "claim": "用人单位自用工之日起建立劳动关系",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
                "model_source": "must-not-survive",
            },
            {
                "claim": "用人单位可以随时解除劳动合同",
                "context_id": "ctx-2",
                "quote": contexts["ctx-2"],
            },
        ],
        "insufficient_context": False,
        "sources": ["must-not-survive"],
    }

    with pytest.raises(AnswerContractError, match="response fields"):
        validate_answer(json.dumps(payload, ensure_ascii=False), contexts)

    recovered = recover_grounded_answer(json.dumps(payload, ensure_ascii=False), contexts)
    serialized = serialize_validated_answer(recovered)

    assert recovered.conclusion == "用人单位自用工之日起建立劳动关系。"
    assert len(recovered.evidence) == 1
    assert "model_source" not in serialized
    assert "sources" not in serialized
    assert validate_answer(serialized, contexts) == recovered


def test_recovery_still_fails_closed_when_no_evidence_is_grounded():
    contexts = {"ctx-1": "用人单位应当按时足额支付劳动报酬。"}
    payload = {
        "conclusion": "用人单位可以随时解除劳动合同。",
        "evidence": [
            {
                "claim": "用人单位可以随时解除劳动合同",
                "context_id": "ctx-1",
                "quote": contexts["ctx-1"],
            }
        ],
        "insufficient_context": False,
    }

    with pytest.raises(AnswerContractError, match="no independently grounded evidence"):
        recover_grounded_answer(json.dumps(payload, ensure_ascii=False), contexts)
