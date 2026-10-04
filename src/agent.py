"""
Agentische Variante der RAG-Pipeline (Tool-Calling).

Unterschied zu rag_pipeline.py:
- rag_pipeline.py: feste Pipeline -> sucht IMMER, genau einmal.
- agent.py: das LLM entscheidet selbst, OB es sucht, WIE OFT und mit
  welchem Filter (z.B. getrennt pro Produkt bei Vergleichsfragen).

CLI-Test:
    python -m src.agent "Deine Frage hier"
"""
import json
import sys

from src.config import CHAT_MODEL, TOP_K, MAX_DISTANCE, get_openai_client
from src.rag_pipeline import get_collection, embed_query

MAX_STEPS = 5  # Guardrail: maximale Anzahl an Agent-Runden
MAX_HISTORY = 6  # letzte 6 Nachrichten (= 3 Frage-Antwort-Paare) als Gespraechskontext

# Guardrail: Text, mit dem das Modell zur Verifikation aufgefordert wird,
# falls es Tools benutzt, aber nie in den Dokumenten gesucht hat.
VERIFY_NUDGE = (
    "[Automatic check] You answered without calling search_documents. "
    "Product variants (e.g. different pens) are often not shown in the document list. "
    "If the question is about product information, call search_documents now before answering. "
    "If the user only asked which documents exist, simply repeat your answer."
)
AGENT_SYSTEM_PROMPT = """You are an internal document assistant for Quality Assurance,
Regulatory Affairs and Logistics in a pharmaceutical company.
You are NOT a clinical tool and do not give medical recommendations.

You have tools to access the company's document collection (drug labels):
- Use search_documents for any question about product information
  (storage, composition, packaging, label versions, manufacturers, ...).
- Use list_documents to see which documents/products exist, or to get the
  exact file name for a filtered search.
- For questions about several products, search separately for each product
  using the source_file filter.
- Never guess file names: if you are not sure, call list_documents first.
- The document list shows only one product name per file, but a document can cover
  several product variants (e.g. different pens). Never conclude that something is
  missing without searching first.
- Include the product name in your search query (e.g. "BASAGLAR storage after opening").
- For greetings or questions about yourself, answer directly without tools.

Rules:
- Answer ONLY based on tool results. If nothing relevant is found, say so honestly.
- Make sure each piece of information refers to the correct product.
- Always answer in the same language as the question.
- Stay faithful to the source: keep conditions exactly as stated (e.g. "when traveling"),
  include all limits (durations, temperatures), and never transfer categories from one product to another.
- If the sources describe products differently, say so instead of forcing a parallel structure.
- End with the documents you used, e.g. "Sources: insulin_glargine.txt (Storage and Handling)"."""

# Beschreibung der Tools fuer das LLM (JSON Schema)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search in the drug label documents. Returns the most "
                "relevant text passages with file name and section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, e.g. 'storage temperature after opening'.",
                    },
                    "source_file": {
                        "type": "string",
                        "description": (
                            "Optional: restrict the search to one document, e.g. "
                            "'insulin_glargine.txt'. Use list_documents to see file names."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "Lists all available documents with product name, label version and effective date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------- Tool-Implementierungen (werden von UNSEREM Code ausgefuehrt) ----------
def known_sources() -> list[str]:
    """Alle Dateinamen, die in der Datenbank existieren."""
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})


def tool_list_documents() -> str:
    data = get_collection().get(include=["metadatas"])
    docs = {}
    for meta in data["metadatas"]:
        docs[meta["source"]] = meta  # ein Eintrag pro Datei
    lines = [
        f"- {src}: {m.get('product')} | label version {m.get('label_version')} | effective {m.get('effective_date')}"
        for src, m in sorted(docs.items())
    ]
    return "Available documents:\n" + "\n".join(lines)


def tool_search_documents(query: str, source_file: str = None):
    """Rueckgabe: (Text fuer das LLM, Liste der verwendeten Treffer)."""
    # 1. Ungueltigen Dateinamen abfangen -> hilfreiche Fehlermeldung (Self-Correction)
    if source_file and source_file not in known_sources():
        return (
            f"Error: unknown source_file '{source_file}'. "
            f"Valid file names are: {', '.join(known_sources())}. Retry with one of these.",
            [],
        )

    kwargs = {"query_embeddings": [embed_query(query)], "n_results": TOP_K}
    if source_file:
        kwargs["where"] = {"source": source_file}  # Metadaten-Filter
    results = get_collection().query(**kwargs)

    hits = []
    for chunk, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # 2. Schwelle nur ohne Filter: mit Filter hat das Modell das Dokument bewusst gewaehlt
        if source_file or dist <= MAX_DISTANCE:
            hits.append((chunk, {**meta, "distance": dist}))

    if not hits:
        return "No relevant passages found for this query.", []

    text = "\n\n".join(
        f"[{m['source']} | {m.get('section')} | chunk {m['chunk_index']}]\n{c}" for c, m in hits
    )
    return text, hits


# ---------- Agent-Loop ----------

def run_agent(question: str, verbose: bool = True, history: list = None):
    """
    Rueckgabe: (answer, retrieved, tool_log)
    - history: bisherige Nachrichten [{"role": "user"/"assistant", "content": "..."}]
    """
    client = get_openai_client()

    # Context Injection: Dokumentliste direkt mitgeben -> Modell muss keine Dateinamen raten
    system_prompt = AGENT_SYSTEM_PROMPT + "\n\n" + tool_list_documents()
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-MAX_HISTORY:])  # Gespraechskontext (begrenzt)
    messages.append({"role": "user", "content": question})

    retrieved, tool_log = [], []
    
    nudged = False  # Guardrail nur einmal ausloesen
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0.2
        )
        msg = response.choices[0].message

        # Kein Tool-Wunsch -> Modell will final antworten
        if not msg.tool_calls:
            used_search = any(t["tool"] == "search_documents" for t in tool_log)

            # Guardrail: Tools benutzt, aber nie gesucht -> einmal zur Verifikation auffordern
            if tool_log and not used_search and not nudged:
                nudged = True
                tool_log.append({"step": step + 1, "tool": "guardrail", "args": {"reason": "no search before answer"}})
                if verbose:
                    print(f"  [Guardrail, Schritt {step + 1}] keine Suche -> Agent wird zur Verifikation aufgefordert")
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": VERIFY_NUDGE})
                continue  # naechste Runde des Loops

            return msg.content, retrieved, tool_log

        # Die Tool-Anfrage des Modells muss im Verlauf bleiben
        messages.append(msg)

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            tool_log.append({"step": step + 1, "tool": name, "args": args})
            if verbose:
                print(f"  [Agent, Schritt {step + 1}] {name}({args})")

            if name == "search_documents":
                result_text, hits = tool_search_documents(
                    query=args.get("query", ""), source_file=args.get("source_file")
                )
                retrieved.extend(hits)
            elif name == "list_documents":
                result_text = tool_list_documents()
            else:
                result_text = f"Unknown tool: {name}"

            # Ergebnis zurueck an das Modell - verknuepft ueber tool_call_id
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})

    return "Abgebrochen: maximale Anzahl an Agent-Schritten erreicht.", retrieved, tool_log


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Benutzung: python -m src.agent "Deine Frage hier"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print("\n=== Agent-Trace ===")
    answer, retrieved, tool_log = run_agent(question)
    if not tool_log:
        print("  (keine Tools verwendet)")

    print("\n=== Antwort ===")
    print(answer)