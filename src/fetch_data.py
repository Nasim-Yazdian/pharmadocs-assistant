"""
Laedt Beispieldaten (Arzneimittel-Labels) von der openFDA Drug Label API
und speichert sie als TXT-Dateien in data/documents/.

Fokus: nicht-klinische Abschnitte, die fuer QA, Regulatory Affairs und
Logistik relevant sind (Zusammensetzung, Lieferform, Lagerung,
Verpackung, Aenderungshistorie).

Ausfuehren mit:
    python -m src.fetch_data
"""
import os
import re
import time
from html.parser import HTMLParser
import requests

from src.config import DOCUMENTS_DIR

API_URL = "https://api.fda.gov/drug/label.json"

# Wirkstoffe, fuer die jeweils das aktuellste Label geladen wird.
# Bewusst gemischt: Tabletten, Suspension, Kuehlware (Insulin, Biologika), Notfall-Injektor.
DRUGS = [
    "metformin",
    "atorvastatin",
    "amoxicillin",
    "omeprazole",
    "levothyroxine",
    "insulin glargine",
    "adalimumab",
    "epinephrine",
]

# (Feldname in openFDA, Ueberschrift in unserer TXT-Datei)
# Bewusst NICHT enthalten: klinische Abschnitte (Dosierung, Warnungen, ...)
# und *_table-Felder (enthalten HTML).
SECTIONS = [
    ("indications_and_usage", "Indications (short context)"),
    ("description", "Description and Composition"),
    ("dosage_forms_and_strengths", "Dosage Forms and Strengths"),
    ("how_supplied", "How Supplied"),
    ("storage_and_handling", "Storage and Handling"),
    ("package_label_principal_display_panel", "Package Label"),
    ("recent_major_changes", "Recent Major Changes"),
]


def _as_text(value) -> str:
    """openFDA liefert Textfelder als Liste von Strings - wir machen einen String daraus."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value) if value else ""


def _first(value) -> str:
    """Fuer Felder wie brand_name: nur der erste Eintrag der Liste."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value else ""


def _clean(text: str) -> str:
    """Mehrfache Leerzeichen/Zeilenumbrueche zu einem Leerzeichen zusammenfassen."""
    return re.sub(r"\s+", " ", text).strip()


def _format_date(yyyymmdd: str) -> str:
    """'20260812' -> '2026-08-12'"""
    if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or "unknown"

class _TableParser(HTMLParser):
    """Sammelt Zeilen (<tr>) und Zellen (<td>/<th>) aus einer HTML-Tabelle."""

    def __init__(self):
        super().__init__()
        self.rows = []      # fertige Zeilen: Liste von Listen
        self._row = []      # aktuelle Zeile
        self._cell = []     # Textstuecke der aktuellen Zelle
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._row.append(_clean(" ".join(self._cell)))
            self._in_cell = False
        elif tag == "tr" and any(self._row):  # leere Zeilen ignorieren
            self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def table_to_text(html: str) -> str:
    """
    Wandelt eine HTML-Tabelle in selbsterklaerende Zeilen um:
    '- <Zeilenname> | <Spaltenname>: <Wert>'
    So bleibt die Zuordnung Wert <-> Spalte auch nach dem Chunking erhalten.
    """
    parser = _TableParser()
    parser.feed(html)
    rows = parser.rows
    if len(rows) < 2:  # keine echte Tabelle -> einfach Text zurueckgeben
        return " ".join(" ".join(r) for r in rows)

    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        row_name = row[0]
        for j in range(1, len(row)):
            column_name = header[j] if j < len(header) else f"Column {j + 1}"
            if row[j]:
                lines.append(f"- {row_name} | {column_name}: {row[j]}")
    return "\n".join(lines)

def slugify(name: str) -> str:
    """'insulin glargine' -> 'insulin_glargine' (sicherer Dateiname)"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch_label(drug: str):
    """Holt das aktuellste Label fuer einen Wirkstoff. Gibt None zurueck, wenn nichts gefunden."""
    params = {
        "search": f'openfda.generic_name:"{drug}" AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
        "sort": "effective_time:desc",
        "limit": 1,
    }
    response = requests.get(API_URL, params=params, timeout=30)
    if response.status_code == 404:  # openFDA meldet "keine Treffer" mit 404
        return None
    response.raise_for_status()  # bei anderen Fehlern (z.B. 429, 500) abbrechen
    return response.json()["results"][0]


def format_label(label: dict) -> str:
    """Wandelt ein openFDA-Label (dict) in einen lesbaren Text um. Ohne Internet testbar."""
    openfda = label.get("openfda", {})
    lines = [
        f"Product: {_first(openfda.get('brand_name')) or 'unknown'}",
        f"Generic name: {_first(openfda.get('generic_name')) or 'unknown'}",
        f"Manufacturer: {_first(openfda.get('manufacturer_name')) or 'unknown'}",
        f"Application number: {_first(openfda.get('application_number')) or 'unknown'}",
        f"Label version: {label.get('version', 'unknown')}",
        f"Effective date: {_format_date(label.get('effective_time', ''))}",
        f"Set ID: {label.get('set_id', 'unknown')}",
        "Source: openFDA Drug Label API (U.S. FDA)",
    ]
    for field, heading in SECTIONS:
        text = _clean(_as_text(label.get(field)))
        if text:  # Abschnitt nur aufnehmen, wenn er im Label existiert
            lines.append("")
            lines.append(f"## {heading}")
            lines.append(text)
            # Falls es zu diesem Abschnitt Tabellen gibt: strukturiert anhaengen
            for table_html in label.get(f"{field}_table", []):
                table_text = table_to_text(table_html)
                if table_text:
                    lines.append("Table (structured):")
                    lines.append(table_text)
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    saved = 0
    for drug in DRUGS:
        print(f"Lade Label fuer '{drug}' ...")
        label = fetch_label(drug)
        if label is None:
            print("  -> nichts gefunden, uebersprungen")
            continue
        path = os.path.join(DOCUMENTS_DIR, f"{slugify(drug)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(format_label(label))
        print(f"  -> gespeichert: {path}")
        saved += 1
        time.sleep(0.5)  # hoeflich zur API: nicht zu schnell hintereinander
    print(f"\nFertig: {saved} von {len(DRUGS)} Dokumenten gespeichert in {DOCUMENTS_DIR}/")


if __name__ == "__main__":
    main()