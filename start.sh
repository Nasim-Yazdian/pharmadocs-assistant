#!/bin/sh
# Startskript fuer den Container:
# 1. Falls die Wissensdatenbank fehlt -> zuerst Ingest (Embeddings erzeugen)
# 2. Dann Streamlit starten
set -e  # bei einem Fehler sofort abbrechen

if [ ! -d chroma_db ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
  echo "[start] Wissensdatenbank fehlt -> starte Ingest ..."
  python -m src.ingest
fi

echo "[start] Starte Streamlit auf Port ${PORT:-8501} ..."
exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0