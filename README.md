# RAG Document Assistant

Ein kleines Retrieval-Augmented-Generation-System, das Fragen zu einer
Sammlung von PDF-Dokumenten beantwortet - inklusive Quellenangabe.

## Architektur

Zwei getrennte Pipelines:

1. **Ingestion (einmalig):** PDF-Dokumente -> Chunking -> Embeddings
   (OpenAI `text-embedding-3-small`) -> Vector-Datenbank (ChromaDB).
2. **Query (pro Frage):** Nutzerfrage -> Embedding der Frage ->
   Retrieval der Top-k relevantesten Chunks aus ChromaDB -> Prompt mit
   Kontext -> Antwort von GPT-4o-mini, inklusive Quellenangabe.

Die Oberflaeche ist mit Streamlit gebaut, die App laeuft containerisiert
mit Docker.

## Lokal ausfuehren

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # dann OPENAI_API_KEY eintragen

# PDFs in data/documents/ ablegen, dann:
python3 -m src.ingest

# Testen ueber die Kommandozeile:
python3 -m src.rag_pipeline "Deine Frage hier"

# Oder als UI:
streamlit run app.py
```

## Mit Docker

```bash
docker build -t rag-portfolio .
docker run -p 8501:8501 --env-file .env rag-portfolio
```

## Live-Demo

<!-- Nach dem Deployment hier den Render/Railway-Link eintragen -->
[Live-Demo](https://your-app.onrender.com)

## Warum dieses Projekt

Gebaut, um praktische Erfahrung mit Retrieval-Augmented Generation und
End-to-End-Deployment zu sammeln: Datenaufbereitung, Vector-Datenbanken,
Prompt-Engineering, Containerisierung und Cloud-Deployment in einem
Projekt kombiniert.
