"""Unit-Tests fuer das Formatieren der openFDA-Daten (ohne Internet)."""
from src.fetch_data import _format_date, format_label, slugify, table_to_text

# Kleine Beispiel-Tabelle im openFDA-Stil (inkl. leerer Zelle <td/> und HTML-Entity &#xB0; = °)
TABLE_HTML = (
    "<table>"
    "<tr><td/><td>Room Temperature (up to 30&#xB0;C)</td><td>Refrigerated</td></tr>"
    "<tr><td>KwikPen</td><td>28 days</td><td>Until expiration date</td></tr>"
    "</table>"
)


def test_table_to_text_keeps_row_and_column_together():
    """Regressionstest fuer das Tabellen-Problem: jeder Wert muss
    zusammen mit Zeilen- und Spaltennamen in einer Zeile stehen."""
    text = table_to_text(TABLE_HTML)
    assert "- KwikPen | Room Temperature (up to 30°C): 28 days" in text
    assert "- KwikPen | Refrigerated: Until expiration date" in text


def test_format_label_header_and_sections():
    label = {
        "openfda": {"brand_name": ["TESTPEN"], "generic_name": ["TESTIN"], "manufacturer_name": ["ACME"]},
        "version": "7",
        "effective_time": "20260812",
        "storage_and_handling": ["Store   cold.\n Do not freeze."],
        "storage_and_handling_table": [TABLE_HTML],
    }
    text = format_label(label)
    assert "Product: TESTPEN" in text
    assert "Effective date: 2026-08-12" in text
    assert "## Storage and Handling" in text
    assert "Store cold. Do not freeze." in text   # Leerzeichen normalisiert
    assert "Table (structured):" in text
    assert "## How Supplied" not in text          # fehlender Abschnitt wird ausgelassen


def test_format_label_missing_fields_do_not_crash():
    text = format_label({})
    assert "Product: unknown" in text


def test_slugify_and_format_date():
    assert slugify("insulin glargine") == "insulin_glargine"
    assert _format_date("20260812") == "2026-08-12"
    assert _format_date("") == "unknown"