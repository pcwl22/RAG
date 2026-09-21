"""Structured, citation-grounded answer contract.

The model is allowed to propose a conclusion and quotes only.  It is not
allowed to choose source names, chapter locations, or arbitrary context IDs;
those are resolved by the server after validation.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_ARTICLE_PATTERN = re.compile(
    r"第[一二三四五六七八九十百千万零〇两0-9]+条"
    r"(?:之[一二三四五六七八九十0-9]+)?"
)
_NUMERIC_FACT_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:万|亿)?(?:元|万元|年|个月|月|日|岁|周岁|人|次|%)?"
)
_LEGAL_OUTCOME_PATTERNS = (
    "死刑",
    "无期徒刑",
    "有期徒刑",
    "拘役",
    "管制",
    "罚金",
    "罚款",
    "没收",
    "赔偿",
    "补偿",
    "违约责任",
    "解除合同",
    "合同无效",
    "行政处罚",
    "刑事责任",
    "民事责任",
    "数罪并罚",
)
_MAX_CONCLUSION_LENGTH = 12000
_MAX_QUOTE_LENGTH = 4000
_MAX_EVIDENCE_ITEMS = 20


class AnswerContractError(ValueError):
    """Raised when an LLM response cannot be grounded in retrieved context."""


@dataclass(frozen=True)
class AnswerEvidence:
    """A validated quote and the context entry from which it came."""

    claim: str
    context_id: str
    quote: str


@dataclass(frozen=True)
class ValidatedAnswer:
    """The only model-produced fields that may reach answer formatting."""

    conclusion: str
    evidence: tuple[AnswerEvidence, ...]
    insufficient_context: bool


def contract_context_id(index: int) -> str:
    """Return a request-local, non-secret context identifier."""
    if index < 1:
        raise ValueError("context indexes are one-based")
    return f"ctx-{index}"


def contract_instructions() -> str:
    """Return strict output instructions shared by normal and aggregate calls."""
    return (
        "只输出一个合法 JSON 对象，不要 Markdown、代码围栏、前后解释或旧版回答标题。"
        "JSON 必须严格包含："
        '{"conclusion":"最终结论","evidence":[{"claim":"结论中的一个完整分句",'
        '"context_id":"ctx-1",'
        '"quote":"上下文中的连续原文片段"}],"insufficient_context":false}。'
        "conclusion 中每个由句号、分号、问号、感叹号或换行分隔的实质结论分句，都必须"
        "有至少一个 evidence 项；claim 必须逐字等于对应的完整结论分句。同一 claim 可用"
        "多个 evidence 项支持，但不得用一个无关 quote 支持另一个结论。"
        "context_id 只能使用本次上下文中实际出现的 ctx-N；quote 必须逐字复制对应上下文的"
        "连续原文子串，不得改写、拼接或补充上下文外内容。每个法条号、事实、法律后果和"
        "适用条件都必须由问题或 evidence 中的原文直接支持。无法得到直接支持时，"
        '将 insufficient_context 设为 true、evidence 设为空数组，并在 conclusion 中明确'
        "说明资料不足；不得在资料不足模式中保留任何 evidence。支持性回答优先使用一至三个"
        "短分句，每个分句尽量沿用其 quote 中的关键措辞；删除证据没有直接覆盖的建议、风险、"
        "程序、例外或扩展判断，不要为了完整或多角度而增加无依据分句。"
    )


def _decode_json(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        raise AnswerContractError("empty model response")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some gateways prepend a short label despite the instruction.  Only
        # recover an object; any free-form answer remains invalid.
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise AnswerContractError("response is not a JSON object") from None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnswerContractError("response contains invalid JSON") from exc


def _validate_article_numbers(conclusion: str, contexts: Mapping[str, str]) -> None:
    context_articles = {
        article for text in contexts.values() for article in _ARTICLE_PATTERN.findall(text)
    }
    unsupported = sorted(set(_ARTICLE_PATTERN.findall(conclusion)) - context_articles)
    if unsupported:
        raise AnswerContractError(
            "conclusion contains article numbers absent from retrieved context"
        )


def _validate_specific_claims(
    conclusion: str,
    contexts: Mapping[str, str],
    query: str,
) -> None:
    """Reject exact numeric or legal-outcome claims absent from support text.

    Natural-language entailment is not reduced to a perfect string check. This
    conservative check nevertheless prevents the most damaging class of
    unsupported additions (amounts, durations, ages, punishments and legal
    consequences) while allowing ordinary connective language and paraphrase.
    """
    context_text = "\n".join(contexts.values())
    supported_text = f"{query}\n{context_text}"
    for literal in _NUMERIC_FACT_PATTERN.findall(conclusion):
        if literal not in supported_text:
            raise AnswerContractError("conclusion contains an unsupported numeric fact")
    for outcome in _LEGAL_OUTCOME_PATTERNS:
        if outcome in conclusion and outcome not in context_text:
            raise AnswerContractError("conclusion contains an unsupported legal consequence")


_CLAIM_SPLIT_PATTERN = re.compile(r"[。！？；;\n]+")
_CLAIM_EDGE_PATTERN = re.compile(
    r"^(?:[-*•]+|(?:\(?[一二三四五六七八九十0-9]+[)）.、：:]\s*))+"
)
_GENERIC_GROUNDING_BIGRAMS = {
    "一个",
    "以及",
    "依据",
    "依法",
    "可以",
    "可能",
    "如果",
    "应当",
    "应予",
    "当前",
    "情形",
    "相关",
    "结论",
    "规定",
    "需要",
}


def _normalize_claim(text: str) -> str:
    normalized = re.sub(r"\s+", "", text).strip("。！？；;，,：:")
    return _CLAIM_EDGE_PATTERN.sub("", normalized)


def _conclusion_claims(conclusion: str) -> list[str]:
    return [
        normalized
        for part in _CLAIM_SPLIT_PATTERN.split(conclusion)
        if (normalized := _normalize_claim(part))
    ]


def _has_lexical_grounding(claim: str, support_text: str) -> bool:
    """Require substantial evidence-specific Chinese/alphanumeric overlap.

    Exact entailment needs a dedicated NLI model and cannot safely be inferred
    from arbitrary retrieved context.  This deliberately conservative lexical
    guard complements the exact checks for article numbers, numeric facts and
    legal outcomes.  Requiring a bounded overlap ratio prevents a generic legal
    phrase or a single shared bigram from grounding an otherwise unrelated
    conclusion while still permitting close paraphrases.
    """
    if _normalize_claim(claim) in re.sub(r"\s+", "", support_text):
        return True

    claim_chars = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", claim))
    support_chars = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", support_text))
    if len(claim_chars) < 2 or len(support_chars) < 2:
        return False
    bigrams = {
        claim_chars[index : index + 2]
        for index in range(len(claim_chars) - 1)
    } - _GENERIC_GROUNDING_BIGRAMS
    if not bigrams:
        return False

    matched = sum(bigram in support_chars for bigram in bigrams)
    required = 1 if len(bigrams) == 1 else max(2, math.ceil(len(bigrams) * 0.35))
    return matched >= required


def validate_answer(
    raw: str,
    contexts: Mapping[str, str],
    query: str = "",
) -> ValidatedAnswer:
    """Parse and validate a model response against request-local context.

    Context IDs and quotes are validated against the exact text shown to the
    model.  The caller must use the returned evidence to build server-side
    source locations; no model-provided source metadata is accepted.
    """
    payload = _decode_json(raw)
    if not isinstance(payload, dict):
        raise AnswerContractError("response root must be an object")
    if set(payload) != {"conclusion", "evidence", "insufficient_context"}:
        raise AnswerContractError("response fields do not match the answer contract")

    conclusion = payload.get("conclusion")
    evidence = payload.get("evidence")
    insufficient_context = payload.get("insufficient_context")
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise AnswerContractError("conclusion must be a non-empty string")
    conclusion = conclusion.strip()
    if len(conclusion) > _MAX_CONCLUSION_LENGTH:
        raise AnswerContractError("conclusion is too long")
    if not isinstance(evidence, list) or len(evidence) > _MAX_EVIDENCE_ITEMS:
        raise AnswerContractError("evidence must be a bounded list")
    if not isinstance(insufficient_context, bool):
        raise AnswerContractError("insufficient_context must be boolean")

    validated: list[AnswerEvidence] = []
    seen: set[tuple[str, str, str]] = set()
    evidence_by_claim: dict[str, list[AnswerEvidence]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise AnswerContractError("each evidence item must be an object")
        if set(item) != {"claim", "context_id", "quote"}:
            raise AnswerContractError("evidence fields do not match the answer contract")
        claim = item.get("claim")
        context_id = item.get("context_id")
        quote = item.get("quote")
        if not isinstance(claim, str) or not _normalize_claim(claim):
            raise AnswerContractError("evidence claim must be non-empty")
        claim = _normalize_claim(claim)
        if not isinstance(context_id, str) or context_id not in contexts:
            raise AnswerContractError("evidence context_id is not in this retrieval result")
        if not isinstance(quote, str) or not quote.strip():
            raise AnswerContractError("evidence quote must be non-empty")
        quote = quote.strip()
        if len(quote) > _MAX_QUOTE_LENGTH:
            raise AnswerContractError("evidence quote is too long")
        if quote not in contexts[context_id]:
            raise AnswerContractError("evidence quote is not an exact context substring")
        key = (claim, context_id, quote)
        if key in seen:
            raise AnswerContractError("duplicate evidence item")
        seen.add(key)
        validated_evidence = AnswerEvidence(claim=claim, context_id=context_id, quote=quote)
        validated.append(validated_evidence)
        evidence_by_claim.setdefault(claim, []).append(validated_evidence)

    if not insufficient_context and not validated:
        raise AnswerContractError("a supported conclusion requires evidence")
    if insufficient_context:
        if validated:
            raise AnswerContractError("an insufficient-context answer must not include evidence")
    else:
        conclusion_claims = _conclusion_claims(conclusion)
        if set(conclusion_claims) != set(evidence_by_claim):
            raise AnswerContractError("every conclusion claim must have its own evidence")
        for claim in conclusion_claims:
            claim_evidence = evidence_by_claim[claim]
            cited_contexts = {
                f"evidence-{index}": item.quote
                for index, item in enumerate(claim_evidence, 1)
            }
            cited_text = "\n".join(cited_contexts.values())
            _validate_article_numbers(claim, cited_contexts)
            _validate_specific_claims(claim, cited_contexts, query)
            if not _has_lexical_grounding(claim, cited_text):
                raise AnswerContractError("claim is not grounded in its cited evidence")
    return ValidatedAnswer(
        conclusion=conclusion,
        evidence=tuple(validated),
        insufficient_context=insufficient_context,
    )


def recover_grounded_answer(
    raw: str,
    contexts: Mapping[str, str],
    query: str = "",
    *,
    max_claims: int = 3,
) -> ValidatedAnswer:
    """Recover only independently valid evidence claims from an invalid response.

    This is deliberately narrower than ``validate_answer``.  It is intended for
    the bounded final fallback after the model has already received one repair
    prompt.  Unknown fields and unsupported conclusion clauses are never
    rendered.  Each retained evidence item must pass the unchanged strict
    validator on its own, and the reconstructed response is validated again as
    a whole before it is returned.
    """

    payload = _decode_json(raw)
    if not isinstance(payload, dict):
        raise AnswerContractError("response root must be an object")

    conclusion = payload.get("conclusion")
    evidence = payload.get("evidence")
    insufficient_context = payload.get("insufficient_context")
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise AnswerContractError("conclusion must be a non-empty string")
    if len(conclusion) > _MAX_CONCLUSION_LENGTH:
        raise AnswerContractError("conclusion is too long")
    if not isinstance(evidence, list) or len(evidence) > _MAX_EVIDENCE_ITEMS:
        raise AnswerContractError("evidence must be a bounded list")
    if not isinstance(insufficient_context, bool):
        raise AnswerContractError("insufficient_context must be boolean")

    if insufficient_context:
        if evidence:
            raise AnswerContractError("an insufficient-context answer must not include evidence")
        sanitized = {
            "conclusion": conclusion.strip(),
            "evidence": [],
            "insufficient_context": True,
        }
        return validate_answer(json.dumps(sanitized, ensure_ascii=False), contexts, query)

    claim_limit = max(1, min(int(max_claims), 3))
    recovered_items: list[dict[str, str]] = []
    recovered_claims: list[str] = []
    seen_claims: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        candidate = {
            "claim": item.get("claim"),
            "context_id": item.get("context_id"),
            "quote": item.get("quote"),
        }
        candidate_claim = candidate["claim"]
        if not isinstance(candidate_claim, str):
            continue
        single = {
            "conclusion": candidate_claim,
            "evidence": [candidate],
            "insufficient_context": False,
        }
        try:
            validated_single = validate_answer(
                json.dumps(single, ensure_ascii=False),
                contexts,
                query,
            )
        except AnswerContractError:
            continue

        grounded = validated_single.evidence[0]
        if grounded.claim in seen_claims:
            continue
        seen_claims.add(grounded.claim)
        recovered_claims.append(grounded.claim)
        recovered_items.append(
            {
                "claim": grounded.claim,
                "context_id": grounded.context_id,
                "quote": grounded.quote,
            }
        )
        if len(recovered_claims) >= claim_limit:
            break

    if not recovered_items:
        raise AnswerContractError("no independently grounded evidence could be recovered")

    reconstructed = {
        "conclusion": "。".join(recovered_claims) + "。",
        "evidence": recovered_items,
        "insufficient_context": False,
    }
    return validate_answer(json.dumps(reconstructed, ensure_ascii=False), contexts, query)


def serialize_validated_answer(answer: ValidatedAnswer) -> str:
    """Serialize a validated answer for cache storage without model-only fields."""

    payload = {
        "conclusion": answer.conclusion,
        "evidence": [
            {
                "claim": item.claim,
                "context_id": item.context_id,
                "quote": item.quote,
            }
            for item in answer.evidence
        ],
        "insufficient_context": answer.insufficient_context,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "AnswerContractError",
    "AnswerEvidence",
    "ValidatedAnswer",
    "contract_context_id",
    "contract_instructions",
    "recover_grounded_answer",
    "serialize_validated_answer",
    "validate_answer",
]
