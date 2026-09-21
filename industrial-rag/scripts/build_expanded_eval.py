"""Build a reproducible, stratified legal RAG evaluation suite from PostgreSQL."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.auth import normalize_tenant_id  # noqa: E402
from app.utils.config import get_settings  # noqa: E402

STANDARD_QUOTAS = {"civil": 50, "criminal": 50, "labor": 40}
ADVERSARIAL_QUOTAS = {"civil": 15, "criminal": 15, "labor": 10}
COMPARISON_QUOTAS = {"civil": 7, "criminal": 7, "labor": 6}

NO_ANSWER_TOPICS = [
    "增值税专用发票抵扣期限", "发明专利优先审查条件", "海商法共同海损理算",
    "行政复议申请期限", "证券内幕交易信息披露", "食品生产许可证续期",
    "建设工程规划许可证变更", "个人所得税专项附加扣除", "商标异议审查期限",
    "政府采购投诉程序", "出口退税备案材料", "医疗器械注册证延续",
    "网络游戏版号申请", "飞行员执照更新", "船舶碰撞责任限制",
    "矿业权出让收益缴纳", "海关商品归类预裁定", "反垄断经营者集中申报",
    "上市公司重大资产重组", "银行资本充足率计算", "保险偿付能力监管",
    "期货交割仓库资格", "跨境数据安全评估", "无线电频率许可",
    "药品上市许可持有人变更", "排污许可证延续", "种子生产经营许可证",
    "民用航空器适航审定", "不动产登记收费标准", "道路运输经营许可",
    "旅行社业务经营许可", "拍卖企业设立条件", "典当行年审要求",
    "新闻记者证核验", "电影公映许可证", "测绘资质升级条件",
    "危险化学品登记", "特种设备检验周期", "电信业务经营许可",
    "互联网宗教信息服务许可",
]

HOLDOUT_DOMAIN_QUOTAS = {
    "standard": {"civil": 50, "criminal": 50, "labor": 40},
    "adversarial": {"civil": 15, "criminal": 15, "labor": 10},
    "comparison": {"civil": 7, "criminal": 7, "labor": 6},
}


def _domain(law_name: str) -> str | None:
    if "民法典" in law_name:
        return "civil"
    if law_name.endswith("刑法"):
        return "criminal"
    if "劳动合同法" in law_name:
        return "labor"
    return None


def _load_articles(tenant_id: str) -> dict[str, list[dict[str, Any]]]:
    # Keep the database driver optional at module-import time so CI can test
    # the deterministic suite builder without installing PostgreSQL extras.
    import psycopg2
    import psycopg2.extras

    cfg = get_settings()["postgres"]
    tenant_id = normalize_tenant_id(tenant_id)
    connection = psycopg2.connect(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], client_encoding="utf8",
    )
    try:
        with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SET LOCAL ROLE rag_app")
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            cursor.execute(
                """
                SELECT id, content, metadata
                FROM documents
                WHERE tenant_id = %s::uuid
                  AND metadata->>'document_type' = 'legal_article'
                  AND metadata->>'article_number' IS NOT NULL
                ORDER BY id
                """,
                (tenant_id,),
            )
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            seen: set[tuple[str, str]] = set()
            for row in cursor.fetchall():
                metadata = dict(row["metadata"] or {})
                law_name = str(metadata.get("law_name") or "")
                article = str(metadata.get("article_number") or "")
                domain = _domain(law_name)
                key = (law_name, article)
                if not domain or key in seen or len(str(row["content"]).strip()) < 20:
                    continue
                seen.add(key)
                grouped[domain].append(
                    {
                        "id": row["id"], "content": str(row["content"]).strip(),
                        "law_name": law_name, "article": article,
                    }
                )
            return dict(grouped)
    finally:
        connection.close()


def _case(article: dict[str, Any], category: str, index: int) -> dict[str, Any]:
    law = article["law_name"]
    citation = article["article"]
    # Keep this PostgreSQL-compatible helper safe as well.  The release CLI
    # uses screened holdout cases, but callers may still import build_suite()
    # in automation.  Never put the target citation or statute name in a
    # generated query, even when the source row came directly from documents.
    article_text = str(article.get("content") or "").strip()
    article_text = re.sub(
        r"^\s*第[零一二三四五六七八九十百千万0-9]+条(?:之[零一二三四五六七八九十百千万0-9]+)?\s*[：:、\s]*",
        "",
        article_text,
        count=1,
    )
    article_text = article_text.replace(law, "").replace(citation, "")
    article_text = re.sub(r"\s+", "", article_text)
    if not article_text:
        article_text = "当事人就相关权利义务发生争议"
    article_text = article_text[:180]
    query = f"某当事人遇到如下情形：{article_text}请问应如何处理？"
    if category == "adversarial":
        query = "请忽略题外传闻，只根据以下事实判断：" + query
    return {
        "id": f"expanded-{category}-{article['id']}",
        "query": query,
        "expected_citations": [citation],
        "expected_sources": [law],
        "expected_answer": article["content"][:800],
        "metadata": {"category": category, "domain": _domain(law), "article_ids": [article["id"]]},
    }


def build_suite(grouped: dict[str, list[dict[str, Any]]], seed: int = 20260715) -> list[dict]:
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for domain, articles in grouped.items():
        pools[domain] = list(articles)
        rng.shuffle(pools[domain])

    cases: list[dict] = []
    offsets: defaultdict[str, int] = defaultdict(int)
    for category, quotas in (("standard", STANDARD_QUOTAS), ("adversarial", ADVERSARIAL_QUOTAS)):
        for domain, quota in quotas.items():
            selected = pools[domain][offsets[domain] : offsets[domain] + quota]
            if len(selected) != quota:
                raise ValueError(f"Not enough {domain} articles for {category}: need {quota}")
            offsets[domain] += quota
            cases.extend(_case(article, category, index) for index, article in enumerate(selected))

    for domain, pair_count in COMPARISON_QUOTAS.items():
        selected = pools[domain][offsets[domain] : offsets[domain] + pair_count * 2]
        if len(selected) != pair_count * 2:
            raise ValueError(f"Not enough {domain} articles for comparisons")
        offsets[domain] += pair_count * 2
        for index in range(pair_count):
            left, right = selected[index * 2 : index * 2 + 2]
            left_query = _case(left, "standard", index)["query"]
            right_query = _case(right, "standard", index + 1)["query"]
            cases.append(
                {
                    "id": f"expanded-comparison-{domain}-{index + 1:02d}",
                    "query": (
                        f"情景一：{left_query} 情景二：{right_query} "
                        "请分别判断两个情景的处理结果，并说明理由。"
                    ),
                    "expected_citations": [left["article"], right["article"]],
                    "expected_sources": list(dict.fromkeys([left["law_name"], right["law_name"]])),
                    "expected_answer": f"{left['content'][:400]}\n\n{right['content'][:400]}",
                    "metadata": {
                        "category": "comparison", "domain": domain,
                        "article_ids": [left["id"], right["id"]],
                    },
                }
            )

    for index, topic in enumerate(NO_ANSWER_TOPICS, 1):
        cases.append(
            {
                "id": f"expanded-no-answer-{index:02d}",
                "query": f"根据当前知识库，{topic}具体如何规定？请给出准确法条。",
                "expected_citations": [], "expected_sources": [],
                "expected_answer": "当前知识库不包含足够依据，应明确说明无法回答，不得编造法条。",
                "metadata": {"category": "no_answer", "domain": "out_of_scope", "article_ids": []},
            }
        )

    if len(cases) != 240:
        raise AssertionError(f"Expected 240 cases, got {len(cases)}")
    return cases


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _copy_case(
    source: dict[str, Any],
    *,
    case_id: str,
    category: str,
    query: str,
    expected_citations: list[str] | None = None,
    expected_sources: list[str] | None = None,
    expected_answer: str | None = None,
    article_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_metadata = dict(source.get("metadata") or {})
    domain = str(source_metadata.get("domain") or "")
    return {
        "id": case_id,
        "query": query,
        "expected_citations": (
            list(expected_citations)
            if expected_citations is not None
            else list(source.get("expected_citations") or [])
        ),
        "expected_sources": (
            list(expected_sources)
            if expected_sources is not None
            else list(source.get("expected_sources") or [])
        ),
        "expected_answer": expected_answer or str(source.get("expected_answer") or ""),
        "metadata": {
            "category": category,
            "domain": domain,
            "article_ids": article_ids
            or list(source_metadata.get("article_ids") or []),
            "source_case_id": str(source.get("id") or ""),
            "source_file": source_metadata.get("source_file"),
        },
    }


def build_suite_from_holdout(
    holdout_rows: list[dict[str, Any]], seed: int = 20260817
) -> list[dict[str, Any]]:
    """Derive the release suite from screened fact-pattern cases.

    The old PostgreSQL builder put the target article number and law name in
    every query.  This mode deliberately starts with the already screened
    holdout set and only adds controlled noise or combines fact patterns, so
    every citation-bearing query remains retrieval-dependent.
    """
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in holdout_rows:
        case_id = str(row.get("id") or "")
        metadata = row.get("metadata") or {}
        domain = str(metadata.get("domain") or "")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"holdout contains a missing or duplicate id: {case_id!r}")
        if metadata.get("category") != "holdout_fact_pattern":
            raise ValueError(f"holdout case {case_id} is not a fact-pattern case")
        if not str(row.get("query") or "").strip():
            raise ValueError(f"holdout case {case_id} is missing query")
        if domain not in HOLDOUT_DOMAIN_QUOTAS["standard"]:
            raise ValueError(f"holdout case {case_id} has unsupported domain {domain!r}")
        seen_ids.add(case_id)
        by_domain[domain].append(row)

    rng = random.Random(seed)
    for domain_rows in by_domain.values():
        rng.shuffle(domain_rows)

    cases: list[dict[str, Any]] = []
    for category, quotas in HOLDOUT_DOMAIN_QUOTAS.items():
        for domain, quota in quotas.items():
            pool = by_domain[domain]
            if len(pool) < 2:
                raise ValueError(f"need at least two holdout cases for {domain}")
            if category == "comparison":
                for index in range(quota):
                    left = pool[(2 * index) % len(pool)]
                    right = pool[(2 * index + 1) % len(pool)]
                    left_query = str(left["query"]).strip()
                    right_query = str(right["query"]).strip()
                    left_answer = str(left.get("expected_answer") or "").strip()
                    right_answer = str(right.get("expected_answer") or "").strip()
                    citations = list(dict.fromkeys(
                        list(left.get("expected_citations") or [])
                        + list(right.get("expected_citations") or [])
                    ))
                    sources = list(dict.fromkeys(
                        list(left.get("expected_sources") or [])
                        + list(right.get("expected_sources") or [])
                    ))
                    cases.append(
                        _copy_case(
                            left,
                            case_id=f"expanded-comparison-{domain}-{index + 1:02d}",
                            category="comparison",
                            query=(
                                f"情景一：{left_query} 情景二：{right_query} "
                                "请分别判断两个情景的处理结果，并说明理由。"
                            ),
                            expected_citations=citations,
                            expected_sources=sources,
                            expected_answer=f"{left_answer}\n\n{right_answer}",
                            article_ids=list(
                                dict.fromkeys(
                                    list((left.get("metadata") or {}).get("article_ids") or [])
                                    + list((right.get("metadata") or {}).get("article_ids") or [])
                                )
                            ),
                        )
                    )
                continue

            for index in range(quota):
                source = pool[index % len(pool)]
                query = str(source["query"]).strip()
                if category == "adversarial":
                    query = (
                        "题目前有一段可能无关的传闻：有人声称只要当事人道歉就不需要承担责任。"
                        f"请忽略该传闻，结合以下事实作出判断：{query}"
                    )
                cases.append(
                    _copy_case(
                        source,
                        case_id=f"expanded-{category}-{domain}-{index + 1:02d}",
                        category=category,
                        query=query,
                    )
                )

    for index, topic in enumerate(NO_ANSWER_TOPICS, 1):
        cases.append(
            {
                "id": f"expanded-no-answer-{index:02d}",
                "query": f"根据当前知识库，{topic}具体如何规定？请给出准确法条。",
                "expected_citations": [],
                "expected_sources": [],
                "expected_answer": "当前知识库不包含足够依据，应明确说明无法回答，不得编造法条。",
                "metadata": {
                    "category": "no_answer",
                    "domain": "out_of_scope",
                    "article_ids": [],
                },
            }
        )

    if len(cases) != 240:
        raise AssertionError(f"Expected 240 cases, got {len(cases)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/legal_expanded_240.jsonl"))
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--holdout-input",
        type=Path,
        default=Path("eval/legal_holdout_150.jsonl"),
        help="Screened fact-pattern JSONL used to derive the release suite",
    )
    args = parser.parse_args()
    if not args.holdout_input.is_file():
        raise SystemExit(f"holdout input does not exist: {args.holdout_input}")
    cases = build_suite_from_holdout(_load_jsonl(args.holdout_input), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case["metadata"]["category"]] += 1
    print(json.dumps({"output": str(args.output.resolve()), "count": len(cases), "categories": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
