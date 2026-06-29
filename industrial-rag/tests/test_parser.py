"""Parser and chunker tests."""
from app.parser.chunk import chunk_text_recursive
from app.parser.document_parser import parse_document
from app.parser.legal_parser import build_legal_article_chunks
from app.parser.parent_child_chunker import ParentChildChunker
from app.parser.parent_child_chunking import chunk_text_parent_child


def test_parse_text_document(tmp_path):
    document = tmp_path / "sample.txt"
    document.write_text("hello\nworld\n", encoding="utf-8")

    assert parse_document(str(document)) == "hello\nworld\n"


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
