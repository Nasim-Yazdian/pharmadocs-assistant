"""Unit-Tests fuer die Ingestion-Logik (ohne API-Key, ohne Internet)."""
from src.ingest import build_context_header, chunk_text, parse_header, split_sections


# ---------- chunk_text ----------

def test_chunk_text_short_text_gives_one_chunk():
    assert chunk_text("Hallo Welt", chunk_size=100, overlap=10) == ["Hallo Welt"]


def test_chunk_text_empty_text_gives_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_chunk_text_overlap():
    text = "abcdefghij" * 10  # 100 Zeichen
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    # Die letzten 5 Zeichen von Chunk 1 == die ersten 5 Zeichen von Chunk 2
    assert chunks[0][-5:] == chunks[1][:5]


def test_chunk_text_no_leftover_chunk_of_pure_overlap():
    """Regressionstest fuer den 'ctural Formula'-Bug:
    Wenn ein Chunk das Textende erreicht, darf kein Rest-Chunk folgen,
    der nur aus Overlap besteht."""
    text = "x" * 1180
    chunks = chunk_text(text, chunk_size=700, overlap=120)
    # 0-700 und 580-1180 -> genau 2 Chunks (alte Version erzeugte einen 3. mit 20 Zeichen)
    assert len(chunks) == 2


# ---------- parse_header ----------

def test_parse_header_reads_key_value_lines():
    text = "Product: BASAGLAR\nGeneric name: INSULIN GLARGINE\nSource: openFDA: label\n\n## X\nbody"
    header = parse_header(text)
    assert header["Product"] == "BASAGLAR"
    assert header["Generic name"] == "INSULIN GLARGINE"
    assert header["Source"] == "openFDA: label"  # nur am ERSTEN ':' getrennt


# ---------- split_sections ----------

def test_split_sections_titles_and_empty_sections():
    text = "Product: A\n\n## Storage\nKeep cold.\n## Empty\n\n## Package\nBox of 5"
    sections = split_sections(text)
    titles = [title for title, _ in sections]
    assert titles == ["Document Info", "Storage", "Package"]  # leerer Abschnitt faellt weg
    assert "Keep cold." in dict(sections)["Storage"]


# ---------- build_context_header ----------

def test_context_header_contains_product_and_section():
    header = build_context_header({"Product": "BASAGLAR", "Generic name": "INSULIN"}, "x.txt", "Storage")
    assert "BASAGLAR" in header
    assert "Storage" in header


def test_context_header_falls_back_to_filename():
    header = build_context_header({}, "doc.pdf", "Full Text")
    assert "doc.pdf" in header