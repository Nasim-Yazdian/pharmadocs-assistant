"""
Zentrale Konfiguration fuer das PharmaDocs-Assistant-Projekt.
Alle "Stellschrauben" des Projekts stehen hier, damit man sie nicht in
mehreren Dateien suchen muss.
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ordner
DOCUMENTS_DIR = "data/documents"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

# Chunking
CHUNK_SIZE = 700        # Zeichen pro Chunk
CHUNK_OVERLAP = 120     # Ueberlappung zwischen aufeinanderfolgenden Chunks

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval
TOP_K = 4  # wie viele Chunks pro Frage abgerufen werden
# Relevanz-Schwelle (ChromaDB-Distanz, kleiner = aehnlicher).
# Aus ersten Tests: relevante Treffer ~0.5-0.95, irrelevante > 1.1.
# Vorlaeufiger Wert - sollte mit einem groesseren Testset validiert werden.
MAX_DISTANCE = 1.0


# Generation
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- OpenAI-Client: lazy + Singleton ---
_client = None


def get_openai_client():
    """
    Erzeugt den OpenAI-Client erst beim ersten echten Bedarf (lazy)
    und gibt danach immer dieselbe Instanz zurueck (Singleton).
    So koennen Module ohne API-Key importiert werden (wichtig fuer Tests/CI).
    """
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY ist nicht gesetzt. Lege eine .env-Datei an "
                "(siehe .env.example) und trage deinen Key ein."
            )
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client