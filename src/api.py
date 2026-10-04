"""
REST-API fuer den PharmaDocs Assistant (FastAPI).
Duenne Schicht: keine Logik hier, nur Ein-/Ausgabe -> die Logik liegt in agent.py.

Lokal starten:
    uvicorn src.api:app --reload
Dann im Browser: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import run_agent
from src.rag_pipeline import get_collection

app = FastAPI(
    title="PharmaDocs Assistant API",
    description=(
        "Interner Dokumenten-Assistent fuer QA, Regulatory Affairs und Logistik "
        "(kein klinisches Einsatzszenario). Datenbasis: openFDA Drug Labels."
    ),
    version="0.1.0",
)


# ---------- Datenmodelle (Pydantic) ----------

class AskRequest(BaseModel):
    question: str = Field(
        ...,  # ... = Pflichtfeld
        min_length=3,
        max_length=1000,
        examples=["How should BASAGLAR KwikPen be stored after first use?"],
    )


class Source(BaseModel):
    source: str
    section: str | None = None
    product: str | None = None
    chunk_index: int | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    tool_calls: list[dict]


# ---------- Endpunkte ----------

@app.get("/health")
def health():
    """Prueft, ob der Dienst laeuft und die Wissensdatenbank erreichbar ist."""
    try:
        return {"status": "ok", "chunks_indexed": get_collection().count()}
    except Exception:
        return {"status": "degraded", "detail": "Wissensdatenbank nicht gefunden - wurde ingest ausgefuehrt?"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Stellt dem Agenten eine Frage und gibt Antwort, Quellen und Tool-Aufrufe zurueck."""
    try:
        answer, retrieved, tool_log = run_agent(request.question, verbose=False)
    except Exception as e:
        print(f"[API] Fehler: {e!r}")  # Details nur ins Server-Log, nicht an den Client
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Verarbeitung der Frage.")

    # Jeden Chunk nur einmal auflisten
    seen, sources = set(), []
    for _, meta in retrieved:
        key = (meta["source"], meta.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            sources.append(Source(**{field: meta.get(field) for field in Source.model_fields}))

    return AskResponse(answer=answer or "", sources=sources, tool_calls=tool_log)