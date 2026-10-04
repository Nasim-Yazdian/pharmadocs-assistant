"""
Unit-Tests fuer die Agent-Logik - OHNE OpenAI und OHNE Datenbank.
Ein Fake-Client spielt vorher festgelegte Modell-Antworten ab (Mocking).
So wird UNSERE Logik (Loop, Guardrail, Verlauf) deterministisch getestet.
"""
import json
from types import SimpleNamespace

import src.agent as agent


# ---------- Hilfsfunktionen: Modell-Antworten im OpenAI-Format nachbauen ----------

def tool_call(name, args=None, call_id="call_1"):
    """Ein Tool-Wunsch des Modells, z.B. tool_call('list_documents')."""
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def model_reply(content=None, tool_calls=None):
    """Eine komplette Modell-Antwort: entweder Text (content) oder Tool-Wuensche."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Ersetzt den OpenAI-Client: gibt bei jedem Aufruf die naechste vorbereitete
    Antwort zurueck und merkt sich, was an das 'Modell' geschickt wurde."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        # gleiche Struktur wie client.chat.completions.create(...)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


def use_fakes(monkeypatch, replies):
    """Ersetzt OpenAI-Client und Tools durch Fakes. Gibt den FakeClient zurueck."""
    client = FakeClient(replies)
    monkeypatch.setattr(agent, "get_openai_client", lambda: client)
    monkeypatch.setattr(agent, "tool_list_documents", lambda: "Available documents:\n- insulin_glargine.txt: BASAGLAR")
    monkeypatch.setattr(
        agent,
        "tool_search_documents",
        lambda query, source_file=None: ("[insulin_glargine.txt] 28 days", [("chunk", {"source": "insulin_glargine.txt"})]),
    )
    return client


# ---------- Tests ----------

def test_greeting_uses_no_tools(monkeypatch):
    client = use_fakes(monkeypatch, [model_reply("Hallo!")])
    answer, retrieved, tool_log = agent.run_agent("Hallo", verbose=False)
    assert answer == "Hallo!"
    assert tool_log == []
    assert len(client.requests) == 1  # nur ein Modell-Aufruf, kein Guardrail


def test_guardrail_forces_search(monkeypatch):
    """Nachbau des Tempo-Pen-Problems: Modell schaut nur die Liste an und will
    ohne Suche antworten -> Guardrail muss eingreifen."""
    use_fakes(monkeypatch, [
        model_reply(tool_calls=[tool_call("list_documents")]),
        model_reply("Not found in the document list."),  # will antworten, ohne zu suchen
        model_reply(tool_calls=[tool_call("search_documents", {"query": "Tempo Pen storage"})]),
        model_reply("28 days at room temperature."),
    ])
    answer, retrieved, tool_log = agent.run_agent("And the Tempo Pen?", verbose=False)
    assert [t["tool"] for t in tool_log] == ["list_documents", "guardrail", "search_documents"]
    assert answer == "28 days at room temperature."
    assert retrieved  # es wurden Quellen gesammelt


def test_guardrail_triggers_only_once(monkeypatch):
    """Frage 'Welche Dokumente gibt es?': nach dem Hinweis darf das Modell
    einfach wieder antworten - kein Endlos-Loop."""
    client = use_fakes(monkeypatch, [
        model_reply(tool_calls=[tool_call("list_documents")]),
        model_reply("Here is the list."),
        model_reply("Here is the list."),
    ])
    answer, _, tool_log = agent.run_agent("Which documents do you have?", verbose=False)
    assert [t["tool"] for t in tool_log] == ["list_documents", "guardrail"]
    assert answer == "Here is the list."
    assert len(client.requests) == 3


def test_history_is_sent_to_model(monkeypatch):
    client = use_fakes(monkeypatch, [model_reply("ok")])
    history = [
        {"role": "user", "content": "How should BASAGLAR be stored?"},
        {"role": "assistant", "content": "Room temperature, 28 days."},
    ]
    agent.run_agent("And the Tempo Pen?", verbose=False, history=history)
    messages = client.requests[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "And the Tempo Pen?"


def test_history_is_limited(monkeypatch):
    client = use_fakes(monkeypatch, [model_reply("ok")])
    history = [{"role": "user", "content": f"Frage {i}"} for i in range(10)]
    agent.run_agent("Neue Frage", verbose=False, history=history)
    # system + MAX_HISTORY Verlaufsnachrichten + neue Frage
    assert len(client.requests[0]["messages"]) == 1 + agent.MAX_HISTORY + 1


def test_agent_stops_after_max_steps(monkeypatch):
    """Guardrail gegen Endlos-Schleifen: Modell will immer weiter suchen."""
    endless = [model_reply(tool_calls=[tool_call("search_documents", {"query": "x"})])] * agent.MAX_STEPS
    client = use_fakes(monkeypatch, endless)
    answer, _, _ = agent.run_agent("Frage", verbose=False)
    assert "Abgebrochen" in answer
    assert len(client.requests) == agent.MAX_STEPS


def test_search_rejects_unknown_file(monkeypatch):
    """Regressionstest: erfundener Dateiname -> hilfreiche Fehlermeldung statt leerem Ergebnis."""
    monkeypatch.setattr(agent, "known_sources", lambda: ["insulin_glargine.txt"])
    text, hits = agent.tool_search_documents("storage", source_file="basaglar_kwikpen.txt")
    assert text.startswith("Error: unknown source_file")
    assert "insulin_glargine.txt" in text  # nennt die gueltigen Namen
    assert hits == []