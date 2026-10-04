"""
RAG-Query-Pipeline: nimmt eine Frage entgegen, holt die relevantesten
Chunks aus ChromaDB und laesst das LLM darauf basierend antworten -
inklusive Quellenangabe.

CLI-Test:
    python -m src.rag_pipeline "Deine Frage hier"
"""
import sys

import chromadb

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
    get_openai_client,
)

SYSTEM_PROMPT = """You are an internal document assistant for Quality Assurance,
Regulatory Affairs and Logistics in a pharmaceutical company.
You are NOT a clinical tool and do not give medical recommendations.

Rules:
- Answer ONLY based on the provided context.
- If the answer is not in the context, say so honestly - do not make anything up.
- Make sure the information refers to the correct product.
- Always answer in the same language as the question.
- End with the sources you actually used, e.g. "Sources: 1, 3"."""


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    response = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def retrieve(query: str, top_k: int = TOP_K):
    """
    Holt die top_k aehnlichsten Chunks.
    Rueckgabe: Liste von (chunk_text, metadata); metadata enthaelt zusaetzlich
    'distance' (kleiner = aehnlicher).
    """
    collection = get_collection()
    results = collection.query(query_embeddings=[embed_query(query)], n_results=top_k)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [(c, {**m, "distance": d}) for c, m, d in zip(chunks, metadatas, distances)]


def build_context(retrieved) -> str:
    blocks = []
    for i, (chunk, meta) in enumerate(retrieved):
        blocks.append(f"[Source {i + 1}: {meta['source']}]\n{chunk}")
    return "\n\n".join(blocks)


def answer_question(query: str):
    retrieved = retrieve(query)
    user_message = f"Context:\n{build_context(retrieved)}\n\nQuestion: {query}"

    response = get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    sources = sorted({
        f"{m.get('product', m['source'])} - {m.get('section', '?')} ({m['source']})"
        for _, m in retrieved
    })
    return answer, sources, retrieved


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Benutzung: python -m src.rag_pipeline "Deine Frage hier"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer, sources, retrieved = answer_question(question)

    print("\n=== Retrieval (Top-K Chunks) ===")
    for i, (_, meta) in enumerate(retrieved):
        print(f"{i + 1}. {meta['source']} | {meta.get('section')} | Distanz {meta['distance']:.3f}")

    print("\n=== Antwort ===")
    print(answer)

    print("\n=== Abgerufene Quellen ===")
    for s in sources:
        print(f"- {s}")