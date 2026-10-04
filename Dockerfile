FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Hinweis: chroma_db/ muss entweder VOR dem Build durch "python -m src.ingest"
# erzeugt worden sein (und wird dann mit COPY . . mitkopiert), oder der
# Ingest-Schritt wird beim Container-Start ausgefuehrt (siehe README).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
