from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.core.exceptions import DocumentLoadError
from src.ingestion.loaders import DocumentLoader, TextCleaner


def test_text_cleaner_normalizes_whitespace():
    dirty_text = "Hello   \t\t world!\n\n\n\nThis is a   test. \u200B"
    cleaned = TextCleaner.clean(dirty_text)
    assert "Hello world!" in cleaned
    assert "This is a test." in cleaned
    assert "\u200B" not in cleaned
    assert "\n\n\n" not in cleaned


def test_text_cleaner_strip_html():
    html_raw = "<html><head><script>alert('x');</script></head><body><h1>Title</h1><p>Main body content.</p></body></html>"
    stripped = TextCleaner.strip_html(html_raw)
    assert "alert" not in stripped
    assert "Title" in stripped
    assert "Main body content." in stripped


def test_document_loader_loads_text_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Line 1\nLine 2 with special chars.")
        f_path = f.name

    try:
        loader = DocumentLoader()
        doc = loader.load_file(f_path)
        assert doc.content == "Line 1\nLine 2 with special chars."
        assert doc.filename.endswith(".txt")
        assert doc.metadata["file_size_bytes"] > 0
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_document_loader_loads_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        p1 = Path(temp_dir) / "doc1.txt"
        p2 = Path(temp_dir) / "doc2.md"
        p1.write_text("Document 1 content", encoding="utf-8")
        p2.write_text("# Document 2 Title\nContent here", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(temp_dir)
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert "doc1.txt" in filenames
        assert "doc2.md" in filenames


def test_document_loader_unsupported_format_raises_error():
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        f.write("Some text")
        f_path = f.name

    try:
        loader = DocumentLoader()
        with pytest.raises(DocumentLoadError):
            loader.load_file(f_path)
    finally:
        Path(f_path).unlink(missing_ok=True)
