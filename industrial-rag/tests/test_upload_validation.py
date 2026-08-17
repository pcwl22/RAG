from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.utils.upload_validation import validate_uploaded_document


def test_upload_validation_rejects_extension_spoofing(tmp_path):
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match PDF"):
        validate_uploaded_document(fake_pdf, "pdf", {})


def test_upload_validation_accepts_minimal_docx_container(tmp_path):
    document = tmp_path / "document.docx"
    with ZipFile(document, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document>正文</document>")

    validate_uploaded_document(document, "docx", {})


def test_upload_validation_rejects_binary_text(tmp_path):
    text_file = tmp_path / "bad.txt"
    text_file.write_bytes(b"text\x00binary")

    with pytest.raises(ValueError, match="binary NUL"):
        validate_uploaded_document(text_file, "txt", {})
