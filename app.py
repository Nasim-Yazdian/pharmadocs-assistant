"""
Streamlit-Oberflaeche fuer den PharmaDocs Assistant (Chat mit dem Agenten).

Lokal starten:
    streamlit run app.py
"""
import streamlit as st

from src.agent import run_agent, tool_list_documents
from src.config import CHAT_MODEL

st.set_page_config(page_title="PharmaDocs Assistant", page_icon="💊", layout="centered")

EXAMPLES = [
    "How should BASAGLAR KwikPen be stored after first use?",
    "Wie muss Adalimumab gelagert werden?",
    "Compare the storage requirements of BASAGLAR KwikPen and adalimumab.",
    "Which company manufactures the omeprazole product and which label version is it?",
]

# ---------- Zustand (bleibt zwischen den Reruns erhalten) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ Über dieses Projekt")
    st.write(
        "Interner Wissens-Assistent für **Qualitätssicherung, Regulatory Affairs "
        "und Logistik**. Ein Agent entscheidet selbst, wann und wo er in den "
        "Dokumenten sucht – jede Antwort mit Quellenangabe."
    )
    st.warning("Kein klinisches Einsatzszenario – keine medizinischen Empfehlungen.")
    st.caption(f"Modell: {CHAT_MODEL} · Daten: openFDA Drug Labels (USA)")

    with st.expander("📚 Verfügbare Dokumente"):
        try:
            st.text(tool_list_documents())
        except Exception:
            st.error("Wissensdatenbank nicht gefunden – bitte zuerst `python -m src.ingest` ausführen.")

    st.subheader("Beispielfragen")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    if st.button("🗑️ Verlauf löschen"):
        st.session_state.messages = []


def render_details(retrieved, tool_log):
    """Zeigt Agent-Schritte und Quellen in einem aufklappbaren Bereich."""
    # Jeden Chunk nur einmal anzeigen
    unique, seen = [], set()
    for chunk, meta in retrieved:
        key = (meta["source"], meta.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            unique.append((chunk, meta))

    with st.expander(f"🔎 Quellen ({len(unique)}) und Agent-Trace"):
        st.markdown("**Agent-Schritte:**")
        if tool_log:
            for t in tool_log:
                st.code(f"Schritt {t['step']}: {t['tool']}({t['args']})", language="text")
        else:
            st.write("Keine Tools verwendet.")

        st.markdown("**Abgerufene Textstellen:**")
        for chunk, meta in unique:
            st.markdown(
                f"**{meta.get('product')} – {meta.get('section')}** · "
                f"`{meta['source']}` · Distanz {meta.get('distance', 0):.2f}"
            )
            st.text(chunk[:500] + ("…" if len(chunk) > 500 else ""))


# ---------- Hauptbereich ----------
st.title("💊 PharmaDocs Assistant")
st.caption("Fragen zu Lagerung, Zusammensetzung, Verpackung und Label-Versionen.")

# Bisherigen Verlauf anzeigen (wird bei jedem Rerun neu gezeichnet)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_details(m.get("retrieved", []), m.get("tool_log", []))

# Neue Frage: entweder aus dem Eingabefeld oder von einem Beispiel-Button
question = st.chat_input("Frage zu den Dokumenten stellen …")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    # Bisherigen Verlauf (nur Text) als Kontext fuer das Modell vorbereiten
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent sucht in den Dokumenten …"):
            try:
                answer, retrieved, tool_log = run_agent(question, verbose=False, history=history)
            except Exception as e:
                print(f"[UI] Fehler: {e!r}")  # Details nur ins Log
                answer, retrieved, tool_log = "⚠️ Es ist ein interner Fehler aufgetreten.", [], []
        st.markdown(answer)
        render_details(retrieved, tool_log)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "retrieved": retrieved, "tool_log": tool_log}
    )