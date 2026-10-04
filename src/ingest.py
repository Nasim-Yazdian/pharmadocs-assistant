"""
Ingestion-Pipeline: liest alle Dokumente (TXT und PDF) aus data/documents,
teilt sie strukturbewusst in Chunks, erzeugt Embeddings und speichert alles
in einer lokalen ChromaDB-Datenbank.

Verbesserungen gegenueber der ersten Version:
- Structure-aware Chunking: zuerst nach Abschnitten ("## Ueberschrift")
  trennen, dann innerhalb eines Abschnitts chunken -> kein Chunk mischt
  zwei Themen.
- Contextual Chunk Header: jeder Chunk beginnt mit Produkt + Abschnitt,
  damit er auch allein (ohne Nachbar-Chunks) verstaendlich ist.

Ausfuehren mit:
    python -m src.ingest
"""
import glob
import os

import chromadb
from pypdf import PdfReader

from src.config import (
    DOCUMENTS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    get_openai_client,
)

HEADER_SECTION = "Document Info"


# ---------- 1. Laden ----------

def load_pdf_text(path: str) -> str:
    """Liest den gesamten Text eines PDFs (Seite fuer Seite) aus."""
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_text(path: str) -> str:
    """Laedt eine TXT- oder PDF-Datei als Text."""
    if path.lower().endswith(".pdf"):
        return load_pdf_text(path)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------- 2. Struktur erkennen ----------

def parse_header(text: str) -> dict:
    """
    Liest die 'Schluessel: Wert'-Zeilen am Anfang einer TXT-Datei
    (bis zur ersten Leerzeile), z.B. {'Product': 'BASAGLAR KwikPen', ...}.
    """
    header = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return header


def split_sections(text: str) -> list[tuple[str, str]]:
    """
    Teilt den Text an Zeilen, die mit '## ' beginnen.
    Rueckgabe: Liste von (Abschnittsname, Abschnittstext).
    Alles vor der ersten Ueberschrift landet im Abschnitt 'Document Info'.
    """
    sections = []
    current_title = HEADER_SECTION
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((current_title, "\n".join(current_lines)))
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_title, "\n".join(current_lines)))
    # leere Abschnitte weglassen
    return [(title, body) for title, body in sections if body.strip()]


# ---------- 3. Chunking ----------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Einfaches, transparentes Chunking auf Zeichenbasis mit Ueberlappung.
    Wird jetzt pro Abschnitt aufgerufen, nicht mehr auf das ganze Dokument.
    """
    chunks = []
    start = 0
    text = " ".join(text.split())  # Leerzeichen/Zeilenumbrueche normalisieren
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):  # Textende erreicht -> kein Rest-Chunk, der nur aus Overlap besteht
            break
        start += chunk_size - overlap
    return chunks


def build_context_header(header: dict, filename: str, section: str) -> str:
    """Kontext-Zeile, die vor jeden Chunk gesetzt wird."""
    product = header.get("Product", filename)
    generic = header.get("Generic name", "unknown")
    return f"[Product: {product} | Generic: {generic} | Section: {section}]"


# ---------- 4. Embeddings ----------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Erzeugt Embeddings in einem Batch-Call (guenstiger als einzeln)."""
    response = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------- Hauptprogramm ----------

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    paths = sorted(
        glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))
        + glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf"))
    )
    if not paths:
        print(f"Keine TXT/PDF-Dateien in {DOCUMENTS_DIR}/ gefunden.")
        return

    all_chunks, all_metadatas, all_ids = [], [], []

    for path in paths:
        filename = os.path.basename(path)
        text = load_text(path)

        if filename.lower().endswith(".txt"):
            header = parse_header(text)
            sections = split_sections(text)
        else:  # PDFs haben keine '## '-Struktur -> ein einziger Abschnitt
            header = {}
            sections = [("Full Text", text)]

        doc_chunk_count = 0
        for section, body in sections:
            context = build_context_header(header, filename, section)
            for chunk in chunk_text(body, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append(f"{context}\n{chunk}")
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": doc_chunk_count,
                    "section": section,
                    "product": header.get("Product", filename),
                    "label_version": header.get("Label version", "unknown"),
                    "effective_date": header.get("Effective date", "unknown"),
                })
                all_ids.append(f"{filename}-{doc_chunk_count}")
                doc_chunk_count += 1

        print(f"{filename}: {len(sections)} Abschnitte -> {doc_chunk_count} Chunks")

    # Datenbank neu aufbauen (alte Collection loeschen -> sauberer Neuaufbau)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    print(f"\nErzeuge Embeddings fuer {len(all_chunks)} Chunks ...")
    batch_size = 100
    for start in range(0, len(all_chunks), batch_size):
        end = start + batch_size
        collection.add(
            documents=all_chunks[start:end],
            embeddings=embed_texts(all_chunks[start:end]),
            metadatas=all_metadatas[start:end],
            ids=all_ids[start:end],
        )
        print(f"  Batch {start}-{min(end, len(all_chunks))} gespeichert")

    print(f"\nFertig: {len(all_chunks)} Chunks indexiert in {CHROMA_DIR}/")


if __name__ == "__main__":
    main()