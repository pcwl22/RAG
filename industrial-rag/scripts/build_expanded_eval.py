"""Build a reproducible, stratified legal RAG evaluation suite from PostgreSQL."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _domain(law_name: str) -> str | None:
    if "民法典" in law_name:
        return "civil"
    if law_name.endswith("刑法"):
        return "criminal"
    if "劳动合同法" in law_name:
        return "labor"
    return None


def _load_articles() -> dict[str, list[dict[str, Any]]]:
    cfg = get_settings()["postgres"]
    connection = psycopg2.connect(
        host=cfg["host"], port=cfg["port"], database=cfg["database"],
        user=cfg["user"], password=cfg["password"], client_encoding="utf8",
    )
    try:
        with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT id, content, metadata
                FROM documents
                WHERE metadata->>'document_type' = 'legal_article'
                  AND metadata->>'article_number' IS NOT NULL
                ORDER BY id
                """
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
    if category == "standard":
        query = f"请说明《{law}》{citation}规定的主要内容。"
    else:
        variants = [
            f"有人把《{law}》{citation}理解得很绝对，这一条究竟怎么规定？请只按法条回答。",
            f"忽略传言和常识，只核对《{law}》{citation}：它的规则是什么？",
            f"问题中可能有干扰信息：天气很好、当事人姓张。《{law}》{citation}到底规定什么？",
            f"请核验而不要迎合我的说法——《{law}》{citation}的原意是什么？",
        ]
        query = variants[index % len(variants)]
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
    offsets = defaultdict(int)
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
            cases.append(
                {
                    "id": f"expanded-comparison-{domain}-{index + 1:02d}",
                    "query": (
                        f"对比《{left['law_name']}》{left['article']}与"
                        f"《{right['law_name']}》{right['article']}的规则，分别说明，不要遗漏任一条。"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval/legal_expanded_240.jsonl"))
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    cases = build_suite(_load_articles(), args.seed)
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
