"""Pruebas de integracion basicas del glosario y modo de seguimiento.

No requieren navegador externo: verifican el HTML servido por FastAPI
junto con los recursos CSS y JavaScript.
"""

import html
import re
from html.parser import HTMLParser

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


class ElementCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.by_id = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.by_id[attributes["id"]] = (tag, attributes)


def load_page():
    response = client.get("/app")
    assert response.status_code == 200
    parser = ElementCollector()
    parser.feed(response.text)
    return response.text, parser.by_id


def test_glossary_has_its_own_navigation_and_panel():
    """La tercera pestaña está conectada al panel correspondiente."""
    _, elements = load_page()

    for name in ("detection", "tracking", "glossary"):
        button_tag, button = elements[f"module-{name}"]
        assert button_tag == "button"
        assert button["aria-controls"] == f"{name}-workspace"
        assert f"{name}-workspace" in elements

    tag, attrs = elements["glossary-workspace"]
    assert tag == "section"
    assert "hidden" in attrs  # No debe aparecer abierto al iniciar.


def test_glossary_explains_processing_techniques():
    """El glosario debe incluir los conceptos de la materia."""
    page, _ = load_page()
    summaries = re.findall(r"<summary\b[^>]*>(.*?)</summary>", page, re.I | re.S)
    terms = {
        html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<span\b[^>]*>.*?</span>", "", summary, flags=re.I | re.S))).strip()
        for summary in summaries
    }

    required = {
        "HSV", "Reflectividad (dBZ)", "Umbralización",
        "Gaussiano", "Mediana", "Promedio (media)",
        "Apertura", "Cierre", "Erosión", "Dilatación",
        "Componentes conectados", "Lucas–Kanade", "Dirección (°)",
    }
    assert required <= terms, f"Faltan conceptos: {sorted(required - terms)}"
    assert len(terms) >= len(required)


def test_tracking_frontend_uses_only_precipitation():
    """No debe ofrecer el modo general en la interfaz."""
    _, elements = load_page()
    tag, field = elements["tracking-mode"]
    assert tag == "input"
    assert field["type"] == "hidden"
    assert field["name"] == "tracking_mode"
    assert field["value"] == "precipitation"


def test_glossary_javascript_and_styles_are_served():
    """El frontend debe servir los recursos que activan la pestaña."""
    js_response = client.get("/static/tracking.js")
    css_response = client.get("/static/styles.css")
    assert js_response.status_code == 200
    assert css_response.status_code == 200

    js = js_response.text
    assert 'activateModule("glossary")' in js
    assert 'glossaryModule.addEventListener("click"' in js
    assert "item.panel.hidden = !active" in js

    css = css_response.text
    assert ".glossary-workspace[hidden]" in css
    assert ".glossary-item" in css
