"""Parser and chunker tests."""
from app.parser.chunk import chunk_text_recursive
from app.parser.document_parser import SUPPORTED_EXTENSIONS, parse_document
from app.parser.legal_parser import build_legal_article_chunks
from app.parser.parent_child_chunker import ParentChildChunker
from app.parser.parent_child_chunking import chunk_text_parent_child


def test_parse_text_document(tmp_path):
    document = tmp_path / "sample.txt"
    document.write_text("hello\nworld\n", encoding="utf-8")

    assert parse_document(str(document)) == "hello\nworld\n"


def test_supported_extensions_match_implemented_parsers():
    assert SUPPORTED_EXTENSIONS == {"pdf", "docx", "xlsx", "txt", "md"}


def test_recursive_chunking_produces_chunks():
    text = ("Article 1. Module test content.\n\nArticle 2. More content.\n\n" * 40).strip()

    chunks = chunk_text_recursive(text, chunk_size=120, chunk_overlap=20)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_parent_child_chunking_shape():
    text = ("Article 1. Parent text.\n\nArticle 2. Child text.\n\n" * 60).strip()

    result = chunk_text_parent_child(text)

    assert "parent_chunks" in result
    assert "child_chunks" in result
    assert result["child_chunks"]


def test_parent_child_chunker_counts_tokens_without_import_side_effects():
    chunker = ParentChildChunker()

    assert chunker.count_tokens("token counting smoke") > 0


def test_legal_article_chunks_include_structured_metadata():
    text = """
中华人民共和国测试法

第一编 总则
第一章 基本规定
第一节 一般规定
第一条 为了测试结构化解析，制定本法。
第二条 测试行为应当依法处理。
""".strip()

    chunks = build_legal_article_chunks(text, "中华人民共和国测试法_20240101.docx")

    assert len(chunks) == 2
    first = chunks[0]
    metadata = first["metadata"]
    assert first["content"].startswith("第一条")
    assert metadata["document_type"] == "legal_article"
    assert metadata["law_name"] == "中华人民共和国测试法"
    assert metadata["law_book"] == "第一编 总则"
    assert metadata["law_chapter"] == "第一章 基本规定"
    assert metadata["law_section"] == "第一节 一般规定"
    assert metadata["article_number"] == "第一条"
    assert metadata["article_text"] == first["content"]
    assert "第一条" in metadata["legal_citation"]


def test_criminal_law_article_metadata_includes_hierarchy_crime_and_keywords():
    text = """
中华人民共和国刑法

第二编 分则
第五章 侵犯财产罪
第二百六十四条 盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金。
""".strip()

    chunks = build_legal_article_chunks(text, "中华人民共和国刑法_20201226.docx")

    assert len(chunks) == 1
    metadata = chunks[0]["metadata"]
    assert metadata["level_1_department"] == "刑法"
    assert metadata["level_2_part"] == "第二编 分则"
    assert metadata["level_3_chapter"] == "第五章 侵犯财产罪"
    assert metadata["level_4_section"] == "无"
    assert metadata["level_5_article"] == "第二百六十四条"
    assert metadata["level_6_crime_name"] == "盗窃罪"
    assert metadata["parent_law_id"] == "刑法_20201226"
    assert "盗窃" in metadata["keywords"]
    assert "数额较大" in metadata["keywords"]
    assert "扒窃" in metadata["keywords"]
    assert metadata["semantic_chunk_id"] == "刑法_分则_侵犯财产罪_盗窃罪_264条"


def test_legal_article_number_suffix_is_preserved_in_semantic_id():
    text = """
中华人民共和国刑法

第二编 分则
第四章 侵犯公民人身权利、民主权利罪
第二百六十二条之一 以暴力、胁迫手段组织残疾人、儿童乞讨的，处三年以下有期徒刑或者拘役，并处罚金。
第二百六十二条之二 组织未成年人进行盗窃、诈骗、抢夺、敲诈勒索等违反治安管理活动的，处三年以下有期徒刑或者拘役，并处罚金。
""".strip()

    chunks = build_legal_article_chunks(text, "中华人民共和国刑法_20201226.docx")

    assert [chunk["metadata"]["article_number"] for chunk in chunks] == [
        "第二百六十二条之一",
        "第二百六十二条之二",
    ]
    assert chunks[0]["metadata"]["semantic_chunk_id"].endswith("262条之一")
    assert chunks[1]["metadata"]["semantic_chunk_id"].endswith("262条之二")
