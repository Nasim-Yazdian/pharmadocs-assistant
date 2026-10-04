FROM python:3.11-slim

WORKDIR /app

# 1. Zuerst nur die Abhaengigkeiten -> Docker-Cache:
#    Pakete werden nur neu installiert, wenn sich requirements.txt aendert.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Dann der Code (Geheimnisse & venv werden per .dockerignore ausgeschlossen)
COPY . .

# Python-Ausgaben sofort ins Log schreiben (wichtig fuer Render-Logs)
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Startskript: baut bei Bedarf die Wissensdatenbank und startet Streamlit
CMD ["sh", "start.sh"]